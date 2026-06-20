"""私人资料智能问答系统 — 文档加载与分块"""
import hashlib
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)
from config import CHUNK_SIZE, CHUNK_OVERLAP

# UnstructuredMarkdownLoader 在 Windows 上依赖较重，提供 TextLoader 回退
try:
    from langchain_community.document_loaders import UnstructuredMarkdownLoader
    def _load_md(path): return UnstructuredMarkdownLoader(path).load()
except ImportError:
    def _load_md(path): return TextLoader(path, encoding="utf-8").load()

LOADER_MAP = {
    ".txt":  lambda p: TextLoader(p, encoding="utf-8").load(),
    ".md":   _load_md,
    ".pdf":  lambda p: PyPDFLoader(p).load(),
    ".docx": lambda p: Docx2txtLoader(p).load(),
}


def load_file(file_path: str) -> list[Document]:
    """按扩展名分派加载器，返回文档列表。"""
    ext = Path(file_path).suffix.lower()
    loader = LOADER_MAP.get(ext)
    if loader is None:
        raise ValueError(f"不支持的文件格式: {ext}，支持: {set(LOADER_MAP.keys())}")
    return loader(file_path)


def split_documents(docs: list[Document]) -> list[Document]:
    """使用 RecursiveCharacterTextSplitter 切分文档。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)


def compute_file_hash(file_path: str) -> str:
    """计算文件的 MD5 哈希值。"""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def process_file(file_path: str) -> tuple[list[Document], str]:
    """加载并切分单个文件，注入元数据。返回 (chunks, file_hash)。"""
    file_hash = compute_file_hash(file_path)
    raw_docs = load_file(file_path)
    chunks = split_documents(raw_docs)
    file_name = Path(file_path).name
    ext = Path(file_path).suffix
    for i, doc in enumerate(chunks):
        doc.metadata.update({
            "source": str(Path(file_path).resolve()),
            "content_hash": file_hash,
            "chunk_index": i,
            "file_type": ext,
            "file_name": file_name,
        })
    return chunks, file_hash


def get_all_documents_from_chroma(collection) -> list[Document]:
    """从 ChromaDB 集合中提取所有已存储的文档块，用于 BM25 索引重建。

    每次启动或文档变更后调用，从持久化的 ChromaDB 读出全量文档，
    作为 BM25 的语料库。对于教学项目的文档规模（几十到几百块），
    全量重建耗时在毫秒级。

    参数:
        collection: ChromaDB 的 collection 对象
    返回:
        所有已索引的 Document 列表
    """
    results = collection.get(include=["documents", "metadatas"])
    docs = []
    documents = results.get("documents") or []
    metadatas = results.get("metadatas") or []
    for i, text in enumerate(documents):
        meta = metadatas[i] if i < len(metadatas) else {}
        docs.append(Document(page_content=text, metadata=meta))
    return docs
