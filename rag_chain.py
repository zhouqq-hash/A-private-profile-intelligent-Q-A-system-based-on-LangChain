"""私人资料智能问答系统 — RAG 问答链（升级版）

新增功能：
  - ChatPromptTemplate：支持对话历史占位符
  - 多轮对话：st_messages_to_langchain_history() 转换消息格式
  - 流式输出：LLM 开启 streaming，chian.stream() 逐字返回
"""
import os
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from config import LLM_MODEL, MEMORY_MAX_TURNS, STREAMING_ENABLED

# ============================================================
# System Prompt（系统指令 + 上下文）
# ============================================================

SYSTEM_PROMPT = """\
你是一位贴心的个人知识助手，帮助用户梳理自己的经历、技能和项目，为撰写实习简历提供素材。

请严格根据下方「参考资料」中的文档内容回答问题。规则如下：
1. 只能使用文档中明确出现的信息，禁止编造。
2. 如果文档中没有相关信息，请直接说："根据你提供的资料，暂时没有找到相关信息。"
3. 回答时，请指出信息来源于哪个文件（文档中已标注[来源: xxx]）。
4. 如果用户询问和简历/实习相关的问题，请主动从文档中提取出可用的经历、技能、成果等素材。
5. 回答要简洁有条理，适合直接参考使用。
6. 如果用户的问题涉及之前对话中提到过的内容（如"第一个项目"、"刚才说的技能"），请结合对话历史来理解指代。

参考资料：
{context}"""

# ============================================================
# ChatPromptTemplate —— 包含对话历史占位符
# ============================================================

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])


# ============================================================
# LLM 客户端
# ============================================================

def get_llm():
    """获取 LLM 实例。streaming 由 STREAMING_ENABLED 控制。

    使用 ChatTongyi 直接调用 DashScope 原生 API，无需 openai 兼容模式。
    """
    return ChatTongyi(
        model=LLM_MODEL,
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        temperature=0.3,
        streaming=STREAMING_ENABLED,
    )


# ============================================================
# 辅助函数
# ============================================================

def format_docs(docs):
    """将检索到的文档列表拼接为上下文字符串。"""
    if not docs:
        return "（暂无相关文档）"
    return "\n\n---\n\n".join(
        f"[来源: {doc.metadata.get('file_name', '未知')}]\n{doc.page_content}"
        for doc in docs
    )


def st_messages_to_langchain_history(st_messages: list[dict]) -> list:
    """将 Streamlit 的聊天消息格式转换为 LangChain 消息对象。

    Streamlit 格式：
        [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    转换后：
        [HumanMessage(...), AIMessage(...)]

    只保留最近 MEMORY_MAX_TURNS 轮对话（一轮 = user + assistant）。
    """
    recent = st_messages[-(MEMORY_MAX_TURNS * 2):]
    history = []
    for msg in recent:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history.append(AIMessage(content=msg["content"]))
    return history


# ============================================================
# 构建 RAG 链
# ============================================================

def build_rag_chain(retriever):
    """构建 RAG 对话链。

    输入：{"question": str, "chat_history": list[BaseMessage]}
    输出：str（LLM 生成的回答）

    链路：
      {context: retriever → format_docs, chat_history: 从输入提取, question: 原样}
      → ChatPromptTemplate → LLM → StrOutputParser
    """
    llm = get_llm()
    chain = (
        {
            "context": (lambda x: x["question"]) | retriever | format_docs,
            "chat_history": lambda x: x.get("chat_history", []),
            "question": lambda x: x["question"],
        }
        | QA_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain
