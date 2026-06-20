"""私人资料智能问答系统 — 配置文件"""
import os
from dotenv import load_dotenv

load_dotenv()

# API
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 模型
LLM_MODEL = "qwen-turbo"  # qwen-plus 免费额度耗尽，临时降级
EMBEDDING_MODEL = "text-embedding-v1"

# ChromaDB
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "personal_docs"

# 分块
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# 检索
RETRIEVAL_K = 4

# 上传暂存
UPLOAD_DIR = "./uploads"

# 支持的文件格式
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}

# ============================================================
# 对话记忆
# ============================================================
MEMORY_MAX_TURNS = 10  # 对话历史保留的最大轮次（一问一答 = 1 轮）

# ============================================================
# 混合检索（Hybrid Search: BM25 + 向量）
# ============================================================
HYBRID_RETRIEVAL_ENABLED = True  # 是否启用混合检索（False 退回纯向量检索）
HYBRID_CANDIDATE_K = 10          # 每路检索器返回的候选数量
RRF_K = 60                        # RRF 融合算法的 k 参数

# ============================================================
# 重排序（Cross-Encoder Reranker）
# ============================================================
RERANK_ENABLED = False                       # 是否启用重排序（需能访问 HuggingFace 或配置镜像）
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"     # Cross-Encoder 模型（支持中文）
RERANK_TOP_N = 4                              # 重排序后保留的文档块数量

# ============================================================
# 流式输出
# ============================================================
STREAMING_ENABLED = True  # 是否逐字输出回答
