# 📚 私人资料智能问答系统 (Personal RAG Q&A System)

基于 LangChain + ChromaDB 的 RAG（检索增强生成）智能问答系统。上传你的简历、项目笔记、课程记录等私人文档，AI 帮你梳理经历、提取技能点，为撰写实习简历提供素材。

## ✨ 功能特性

### 文档管理
- **多格式支持** — PDF / Word (.docx) / Markdown / TXT
- **智能去重** — 基于 MD5 哈希的文档去重与增量更新
- **持久化存储** — ChromaDB 向量索引持久化，一次索引多次使用

### 智能检索
- **混合检索** — BM25 关键词 + 向量语义检索，RRF 融合排序
- **重排序** — CrossEncoder (BGE-Reranker) 对候选文档二次精选
- **可配置降级** — 通过配置开关灵活切换检索策略

### 对话体验
- **多轮对话** — 自动注入聊天历史，支持指代追问（"第一个项目用了什么技术？"）
- **流式输出** — 逐字渲染回答，无需等待全部生成
- **防幻觉** — Prompt 约束仅基于文档作答，超出范围主动拒答
- **来源标注** — 每个回答指出信息来源于哪个文件

### Web 界面
- Streamlit 侧栏文档管理（上传 / 索引 / 重置）
- 系统状态面板（当前检索模式、记忆轮次等）
- 一键清空对话 / 重置索引

## 🛠 技术栈

| 组件 | 技术 |
|------|------|
| 框架 | LangChain (LCEL) |
| LLM | 通义千问 (qwen-turbo / qwen-plus) |
| Embedding | DashScope text-embedding-v1 |
| 向量数据库 | ChromaDB |
| 关键词检索 | BM25 (langchain_community) |
| 重排序 | BAAI/bge-reranker-v2-m3 (CrossEncoder) |
| UI | Streamlit |
| 文档解析 | PyPDF · docx2txt · Unstructured |

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/zhouqq-hash/A-private-profile-intelligent-Q-A-system-based-on-LangChain.git
cd A-private-profile-intelligent-Q-A-system-based-on-LangChain
```

### 2. 创建虚拟环境并安装依赖

```bash
python -m venv .venv
source .venv/Scripts/activate  # Git Bash / Linux
# 或
.venv\Scripts\activate         # CMD / PowerShell

pip install -r requirements.txt
```

### 3. 配置 API Key

在项目根目录创建 `.env` 文件：

```
DASHSCOPE_API_KEY=your_dashscope_api_key_here
```

> 💡 前往 [阿里云百炼平台](https://bailian.console.aliyun.com/) 开通 DashScope 服务获取 API Key。

### 4. 启动应用

```bash
streamlit run app.py
```

浏览器会自动打开 `http://localhost:8501`。

## 📖 使用流程

1. **上传文档** — 在侧边栏选择你的简历、项目笔记、课程报告等文件
2. **构建索引** — 点击「🔨 索引文件」，系统自动解析文档并构建向量索引
3. **开始提问** — 在聊天框输入问题，AI 基于你的文档内容作答
4. **多轮追问** — 支持指代追问，如"第一个项目用了什么技术？"

### 示例问题

- "我做过哪些项目？"
- "列出我的所有技能"  
- "根据我的经历，适合投递什么岗位？"
- "帮我整理一段自我介绍"
- "我的项目中最有亮点的是什么？"

## 📁 项目结构

```
├── app.py               # Streamlit Web 界面（支持流式输出 + 多轮对话）
├── config.py            # 集中配置（模型 / 检索 / 记忆 / 分块参数）
├── document_loader.py   # 多格式文档加载 + 文本分块 + MD5 去重
├── vector_store.py      # ChromaDB 管理 + 混合检索 + 重排序
├── rag_chain.py         # RAG 对话链（ChatPromptTemplate + 聊天历史）
├── requirements.txt     # 依赖清单
├── demo/                # 演示文档（示例数据）
│   └── sample_resume.md # 示例简历，可直接上传测试
└── uploads/             # 上传文件暂存（gitignore）
```

## ⚙️ 配置说明

`config.py` 中的关键配置项：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `LLM_MODEL` | `qwen-turbo` | LLM 模型（免费额度耗尽后可切换 qwen-plus） |
| `HYBRID_RETRIEVAL_ENABLED` | `True` | 是否启用 BM25+向量混合检索 |
| `RERANK_ENABLED` | `False` | 是否启用 CrossEncoder 重排序（需下载模型） |
| `STREAMING_ENABLED` | `True` | 是否逐字流式输出 |
| `MEMORY_MAX_TURNS` | `10` | 对话历史保留轮次 |
| `CHUNK_SIZE` | `500` | 文档分块大小（字符） |

## 🎬 Demo 演示

### 在线体验

启动后，使用 `demo/` 目录中的示例文档快速体验：

```bash
# 启动应用
streamlit run app.py

# 在侧边栏上传 demo/sample_resume.md
# 点击「索引文件」后即可开始提问
```

### 示例文档

`demo/sample_resume.md` 包含一份模拟的计算机专业学生简历，涵盖：
- 教育背景
- 项目经历（3 个典型项目）
- 技能清单（编程语言 / 框架 / 工具）
- 实习经历
- 获奖情况

### 效果展示

提问示例：

```
用户: 我做过哪些项目？
AI:   根据你的资料，你参与过以下3个项目：
      1. 基于深度学习的图像分类系统（来源: sample_resume.md）
      2. 在线商城全栈开发（来源: sample_resume.md）
      3. 数据分析与可视化大屏（来源: sample_resume.md）
      ...

用户: 第一个项目用了什么技术？
AI:   你的「基于深度学习的图像分类系统」使用了以下技术：
      - PyTorch 深度学习框架
      - ResNet-50 预训练模型
      - Flask Web 框架搭建 API
      ...

用户: 根据我的经历，适合投递什么岗位？
AI:   根据你的技能和项目经历，建议投递：
      1. 后端开发工程师 — 你的 Flask、Django、MySQL 经验匹配
      2. 数据分析工程师 — 你的 Pandas、ECharts 项目经验匹配
      3. AI/机器学习实习生 — 你的 PyTorch 项目经验匹配
      ...
```

## 📝 依赖清单

参见 `requirements.txt`，主要依赖：

- `streamlit` — Web 界面
- `langchain` / `langchain-core` / `langchain-community` — LangChain 框架
- `langchain-chroma` — ChromaDB 向量存储集成
- `langchain-classic` — 混合检索 & 重排序组件
- `chromadb` — 向量数据库
- `dashscope` — 通义千问 API SDK
- `pypdf` / `docx2txt` — 文档解析
- `sentence-transformers` — 重排序模型（可选）

## 📄 License

MIT License
