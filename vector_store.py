"""私人资料智能问答系统 — ChromaDB 向量存储管理（升级版）

新增功能：
  - 混合检索：BM25 关键词 + 向量语义，RRF 融合
  - 重排序：CrossEncoder 对候选文档二次打分精选
  - 检索缓存：文档变更时自动失效
  - 修复 DashScopeEmbeddings 单字符串输入问题
"""
from pathlib import Path
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from config import (
    CHROMA_PERSIST_DIR, COLLECTION_NAME, EMBEDDING_MODEL, RETRIEVAL_K,
    HYBRID_RETRIEVAL_ENABLED, HYBRID_CANDIDATE_K, RRF_K,
    RERANK_ENABLED, RERANK_MODEL, RERANK_TOP_N,
)
from typing import List, TYPE_CHECKING

# 类型标注专用（不触发实际导入）
if TYPE_CHECKING:
    from langchain_community.retrievers import BM25Retriever
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder
    from langchain_classic.retrievers import ContextualCompressionRetriever


# ============================================================
# 修复版 Embeddings：确保 texts 始终以数组形式传入 DashScope API
# ============================================================

class FixedDashScopeEmbeddings(DashScopeEmbeddings):
    """修复版 DashScope 嵌入模型。

    langchain_community 的 embed_with_retry 存在参数传递问题，
    在特定条件下会触发 DashScope API 的 400 错误。
    此子类直接使用 DashScope SDK，绕过有问题的包装逻辑。
    """

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """直接调用 DashScope SDK，确保参数格式正确。"""
        from dashscope import TextEmbedding
        result = []
        batch_size = 25  # text-embedding-v1 的批处理大小
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            resp = TextEmbedding.call(
                model=self.model,
                input=batch,
                text_type="document",
            )
            if resp.status_code == 200:
                result.extend([item["embedding"] for item in resp.output["embeddings"]])
            else:
                raise ValueError(
                    f"DashScope embedding 失败: status={resp.status_code}, "
                    f"code={resp.code}, message={resp.message}"
                )
        return result

    def embed_query(self, text: str) -> List[float]:
        """直接调用 DashScope SDK，单条查询。"""
        from dashscope import TextEmbedding
        resp = TextEmbedding.call(
            model=self.model,
            input=text,
            text_type="query",
        )
        if resp.status_code == 200:
            return resp.output["embeddings"][0]["embedding"]
        else:
            raise ValueError(
                f"DashScope embedding 失败: status={resp.status_code}, "
                f"code={resp.code}, message={resp.message}"
            )

# ============================================================
# 全局缓存：避免每次查询都重建
# ============================================================
_bm25_retriever = None  # type: ignore — BM25Retriever | None
_reranker_model = None  # type: ignore — HuggingFaceCrossEncoder | None
_compression_retriever = None  # type: ignore — ContextualCompressionRetriever | None


# ============================================================
# 基础组件（不变）
# ============================================================

def get_embeddings():
    """获取 DashScope 嵌入模型（使用修复版，解决单字符串输入问题）。"""
    return FixedDashScopeEmbeddings(model=EMBEDDING_MODEL)


def get_collection() -> Chroma:
    """获取 ChromaDB 集合。"""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_PERSIST_DIR,
    )


# ============================================================
# BM25 关键词检索 —— 从 ChromaDB 重建索引
# ============================================================

def _rebuild_bm25_index():
    """从 ChromaDB 全量读出文档，重建 BM25 索引。

    调用时机：(1) 应用启动  (2) 文档入库后  (3) 索引重置后
    教学项目数据量小（<1000 块），全量重建耗时可忽略。
    """
    global _bm25_retriever
    from langchain_community.retrievers import BM25Retriever  # 懒加载
    from document_loader import get_all_documents_from_chroma
    collection = get_collection()
    all_docs = get_all_documents_from_chroma(collection)
    if all_docs:
        _bm25_retriever = BM25Retriever.from_documents(
            all_docs,
            k=HYBRID_CANDIDATE_K,
        )
    else:
        _bm25_retriever = None


def _get_bm25_retriever():
    """懒加载获取 BM25 检索器（首次调用时自动构建）。"""
    global _bm25_retriever
    if _bm25_retriever is None:
        _rebuild_bm25_index()
    return _bm25_retriever


# ============================================================
# 混合检索器 —— BM25 + 向量 + RRF 融合
# ============================================================

def _get_hybrid_retriever():
    """构建 EnsembleRetriever：向量检索 + BM25 关键词检索，RRF 融合排序。

    每路返回 HYBRID_CANDIDATE_K 个候选，
    通过 RRF（Reciprocal Rank Fusion）合并去重，
    返回综合排序后的候选列表。
    """
    from langchain_classic.retrievers import EnsembleRetriever  # 懒加载
    collection = get_collection()
    vector_retriever = collection.as_retriever(
        search_type="similarity",
        search_kwargs={"k": HYBRID_CANDIDATE_K},
    )
    bm25 = _get_bm25_retriever()

    if bm25 is None or not HYBRID_RETRIEVAL_ENABLED:
        # BM25 无数据或被禁用时，退回纯向量检索
        return vector_retriever

    ensemble = EnsembleRetriever(
        retrievers=[vector_retriever, bm25],
        weights=[0.5, 0.5],  # 等权重 = RRF
        c=RRF_K,
    )
    return ensemble


# ============================================================
# 重排序器 —— CrossEncoder 单例
# ============================================================

def _get_reranker_model():
    """懒加载 CrossEncoder 模型（首次下载后缓存到本地）。

    模型约 2.2GB，首次运行需从 HuggingFace 下载。
    国内用户可设置环境变量 HF_ENDPOINT=https://hf-mirror.com 加速。
    如果下载失败（网络不通），自动降级为 None，跳过重排序。
    """
    global _reranker_model
    if _reranker_model is None:
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder  # 懒加载
        print(f"[Loading] 正在加载重排序模型 {RERANK_MODEL}...（首次可能需要下载）")
        try:
            _reranker_model = HuggingFaceCrossEncoder(
                model_name=RERANK_MODEL,
                model_kwargs={"device": "cpu"},
            )
            print("[OK] 重排序模型加载完成")
        except Exception as e:
            print(f"[WARN] 重排序模型加载失败: {e}")
            print("[WARN] 将跳过重排序，使用纯检索模式。")
            print("[TIP] 国内用户可设置: set HF_ENDPOINT=https://hf-mirror.com")
            _reranker_model = None  # 标记为失败，不再重试
    return _reranker_model


def _invalidate_cache():
    """文档变更时，使缓存的检索器失效。"""
    global _compression_retriever
    _compression_retriever = None


# ============================================================
# 最终检索器 —— 混合检索 + 重排序 → 对外唯一接口
# ============================================================

def get_retriever(k: int = RETRIEVAL_K):
    """返回最终的检索器，供 rag_chain 使用。

    数据流：
      Hybrid(向量+BM25, Top10) → CrossEncoder Rerank → Top4

    通过配置开关可灵活降级：
      - RERANK_ENABLED=False → 跳过重排序
      - HYBRID_RETRIEVAL_ENABLED=False → 退回纯向量检索
    """
    global _compression_retriever
    if _compression_retriever is not None:
        return _compression_retriever

    hybrid_retriever = _get_hybrid_retriever()

    if RERANK_ENABLED:
        from langchain_classic.retrievers import ContextualCompressionRetriever  # 懒加载
        from langchain_classic.retrievers.document_compressors import CrossEncoderReranker  # 懒加载
        reranker = _get_reranker_model()
        if reranker is not None:
            compressor = CrossEncoderReranker(
                model=reranker,
                top_n=RERANK_TOP_N,
            )
            _compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=hybrid_retriever,
            )
            return _compression_retriever

    # 未启用重排序时，直接返回混合检索器
    return hybrid_retriever


# ============================================================
# 文档增删改 —— 核心逻辑不变，增加缓存失效
# ============================================================

def delete_by_source(collection: Chroma, source: str) -> int:
    """删除指定源文件的所有 chunk，返回删除数量。"""
    results = collection.get(where={"source": source})
    ids = results.get("ids", [])
    if ids:
        collection.delete(ids=ids)
    return len(ids)


def upsert_file(collection: Chroma, chunks: list[Document],
                source: str, content_hash: str) -> dict:
    """插入或更新文件 chunk，基于 content_hash 去重。

    返回 {"action": "added"|"skipped"|"updated", "chunks": int}
    """
    existing = collection.get(where={"source": source})
    existing_ids = existing.get("ids", [])
    if existing_ids:
        existing_meta = existing.get("metadatas", [])
        if existing_meta and existing_meta[0].get("content_hash") == content_hash:
            return {"action": "skipped", "chunks": 0}
        collection.delete(ids=existing_ids)
        action = "updated"
    else:
        action = "added"
    collection.add_documents(chunks)
    return {"action": action, "chunks": len(chunks)}


def index_files(file_paths: list[str]) -> dict:
    """批量索引文件。入库后自动重建 BM25 索引。"""
    from document_loader import process_file

    collection = get_collection()
    summary = {"added": [], "skipped": [], "updated": [], "total_chunks": 0}

    for fp in file_paths:
        chunks, fhash = process_file(fp)
        result = upsert_file(collection, chunks, str(Path(fp).resolve()), fhash)
        summary[result["action"]].append(Path(fp).name)
        summary["total_chunks"] += result["chunks"]

    # 文档变更后重建 BM25 并失效检索缓存
    _rebuild_bm25_index()
    _invalidate_cache()
    return summary


def get_indexed_files() -> list[str]:
    """返回已索引的源文件列表。"""
    collection = get_collection()
    results = collection.get()
    sources = set()
    for meta in (results.get("metadatas") or []):
        if meta and "source" in meta:
            sources.add(meta["source"])
    return sorted(sources)


def reset_collection() -> int:
    """清空集合中所有 chunk，并重建 BM25 索引。"""
    collection = get_collection()
    results = collection.get()
    ids = results.get("ids", [])
    if ids:
        collection.delete(ids=ids)
    _rebuild_bm25_index()
    _invalidate_cache()
    return len(ids)
