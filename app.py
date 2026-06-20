"""私人资料智能问答系统 — Streamlit Web UI（升级版）

新增功能：
  - 多轮对话：自动将聊天历史注入 LLM，支持指代追问
  - 流式输出：逐字渲染回答，不再等待全部生成
  - 配置指示：侧栏展示当前检索模式
"""
import streamlit as st
import os
from pathlib import Path
from config import (
    UPLOAD_DIR, SUPPORTED_EXTENSIONS,
    STREAMING_ENABLED, HYBRID_RETRIEVAL_ENABLED, RERANK_ENABLED,
    HYBRID_CANDIDATE_K, RERANK_TOP_N, MEMORY_MAX_TURNS,
)
from vector_store import (
    index_files, get_retriever, get_indexed_files, reset_collection,
)
from rag_chain import build_rag_chain, st_messages_to_langchain_history

st.set_page_config(
    page_title="私人资料智能问答",
    page_icon="📚",
    layout="wide",
)

st.title("📚 私人资料智能问答系统")
st.caption("上传你的简历、项目笔记、课程记录等文档，AI 帮你梳理经历，准备实习简历素材。")

# 确保上传目录存在
Path(UPLOAD_DIR).mkdir(exist_ok=True)

# ============================================================
# 初始化 session state
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = None
if "docs_indexed" not in st.session_state:
    st.session_state.docs_indexed = False


# ============================================================
# 侧栏：文档管理 + 系统状态
# ============================================================
with st.sidebar:
    st.header("📁 文档管理")

    uploaded_files = st.file_uploader(
        "上传文档（支持 PDF / Word / Markdown / TXT）",
        type=list({ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS}),
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("🔨 索引文件", type="primary", use_container_width=True):
        saved_paths = []
        for uf in uploaded_files:
            dest = os.path.join(UPLOAD_DIR, uf.name)
            with open(dest, "wb") as f:
                f.write(uf.getbuffer())
            saved_paths.append(dest)

        with st.spinner("正在解析文档并构建索引..."):
            summary = index_files(saved_paths)

        added_n = len(summary["added"])
        updated_n = len(summary["updated"])
        skipped_n = len(summary["skipped"])

        parts = []
        if added_n:
            parts.append(f"新增 {added_n} 个")
        if updated_n:
            parts.append(f"更新 {updated_n} 个")
        if skipped_n:
            parts.append(f"跳过 {skipped_n} 个（未变化）")
        st.success(f"索引完成！{'，'.join(parts)}，共 {summary['total_chunks']} 个文本块。")

        # 重建链（检索器需要刷新）
        st.session_state.chain = build_rag_chain(get_retriever())
        st.session_state.docs_indexed = True

    st.divider()

    # 已索引文件列表
    indexed = get_indexed_files()
    st.caption(f"已索引文件：{len(indexed)} 个")
    if indexed:
        for f in indexed[:20]:
            st.caption(f"  • {Path(f).name}")
        if len(indexed) > 20:
            st.caption(f"  ... 及其他 {len(indexed) - 20} 个文件")

    st.divider()

    # 操作按钮
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑 重置索引", use_container_width=True):
            count = reset_collection()
            st.warning(f"已清除 {count} 个文本块，索引已重置。")
            st.session_state.messages = []
            st.session_state.chain = None
            st.session_state.docs_indexed = False
            st.rerun()
    with col2:
        if st.button("🧹 清空对话", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.divider()

    # 系统状态指示
    st.caption("⚙️ 系统配置")
    mode_parts = []
    mode_parts.append("🔍 混合检索" if HYBRID_RETRIEVAL_ENABLED else "🔍 纯向量检索")
    mode_parts.append("📊 重排序" if RERANK_ENABLED else "📊 无重排序")
    mode_parts.append("⚡ 流式输出" if STREAMING_ENABLED else "📦 整段输出")
    for p in mode_parts:
        st.caption(f"  {p}")
    st.caption(f"  候选: {HYBRID_CANDIDATE_K} → 精选: {RERANK_TOP_N}")
    st.caption(f"  记忆: 最近 {MEMORY_MAX_TURNS} 轮")


# ============================================================
# 主区域：初始化链
# ============================================================
if st.session_state.chain is None:
    try:
        st.session_state.chain = build_rag_chain(get_retriever())
        st.session_state.docs_indexed = len(get_indexed_files()) > 0
    except Exception as e:
        st.info("👈 请在侧边栏上传文档并点击「索引文件」开始使用。")


# ============================================================
# 主区域：显示聊天历史
# ============================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ============================================================
# 主区域：聊天输入 + 流式生成
# ============================================================
if prompt := st.chat_input("输入你的问题，比如「我做过哪些项目？」支持多轮追问"):
    # 1. 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. AI 回复
    with st.chat_message("assistant"):
        if st.session_state.chain is None:
            response = "请先在侧边栏上传并索引文档，然后我才能回答你的问题。"
            st.markdown(response)
        else:
            # 准备聊天历史（不包含当前问题）
            chat_history = st_messages_to_langchain_history(
                st.session_state.messages[:-1]
            )
            chain_input = {
                "question": prompt,
                "chat_history": chat_history,
            }

            if STREAMING_ENABLED:
                # ---- 流式输出：逐字渲染 ----
                with st.spinner("思考中..."):
                    try:
                        stream_gen = st.session_state.chain.stream(chain_input)
                        response = st.write_stream(stream_gen)
                    except Exception as e:
                        response = f"出错了：{str(e)}"
                        st.error(response)
            else:
                # ---- 非流式输出：整段返回 ----
                with st.spinner("思考中..."):
                    try:
                        response = st.session_state.chain.invoke(chain_input)
                        st.markdown(response)
                    except Exception as e:
                        response = f"出错了：{str(e)}"
                        st.error(response)

    # 3. 记录 AI 回答
    st.session_state.messages.append({"role": "assistant", "content": response})
