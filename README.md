# 明日方舟攻略助手 (Arknights Assistant)

一个基于 RAG（Retrieval-Augmented Generation）+ 多类型记忆系统的智能问答系统，专为《明日方舟》玩家提供准确、专业的游戏攻略和知识查询服务。

## 功能特性

- **智能问答**：基于大语言模型的智能对话，提供准确的游戏攻略
- **知识库管理**：支持自定义知识库的上传和更新（TXT 格式）
- **混合检索**：语义相似度 + BM25 关键词检索，融合排序后返回最优结果
- **语义文本分割**：基于句子边界和语义相似度的智能文本分块
- **多类型记忆系统**：工作记忆、情景记忆、语义记忆、感知记忆四层架构，支持记忆的添加/检索/遗忘/固化
- **对话历史**：基于记忆系统持久化对话上下文，提供更连贯的交互体验
- **Web 界面**：基于 Streamlit 的友好用户界面
- **MD5 去重**：自动检测重复上传，避免知识库冗余

## 技术栈

### 核心框架

- **LangChain** — LLM 应用开发框架
- **Streamlit** — Web 界面构建
- **ChromaDB** — 向量数据库（主知识库）

### AI 模型

- **通义千问 (Qwen3-Max)** — 对话生成模型
- **DashScope Embeddings (text-embedding-v4)** — 文本嵌入模型

### 记忆系统

记忆系统采用四层架构设计，通过 `memory/MemoryTool` 统一对外接口：

| 类型 | 说明 | 存储后端 |
|------|------|----------|
| 工作记忆 (Working) | 短期记忆，TTL 自动过期 | 内存 |
| 情景记忆 (Episodic) | 长期对话记录，持久化存储 | SQLite |
| 语义记忆 (Semantic) | 知识图谱关系，可选启用 | Neo4j |
| 感知记忆 (Perceptual) | 向量化记忆，可选启用 | Qdrant |

四大操作：**add**（添加）、**search**（检索）、**forget**（遗忘）、**consolidate**（固化）

### 检索增强

- **语义检索**：基于 ChromaDB 向量相似度搜索，支持相似度阈值过滤
- **关键词检索**：基于 BM25 算法 + jieba 中文分词
- **混合检索**：语义与关键词结果归一化后加权融合（可配权重），MD5 去重后返回 top-k

### 其他依赖

- Python 3.13+
- jieba — 中文分词
- python-dotenv — 环境变量管理
- Neo4j / Qdrant — 可选记忆后端（云服务）

## 项目结构

```
Arknights_assistant/
├── app_qa.py              # 问答系统主应用（Streamlit）
├── app_file_upload.py     # 知识库上传应用（Streamlit）
├── rag.py                 # RAG 服务核心逻辑
├── knowledge_base.py      # 知识库管理服务（含语义分割器）
├── vector_stores.py       # 向量存储封装（含 BM25 + 混合检索器）
├── file_history_store.py  # 聊天历史存储（桥接记忆系统）
├── config_data.py         # 项目配置文件
├── .env                   # 环境变量配置（API Key 等）
├── md5_txt                # MD5 去重记录文件
├── data/                  # 知识库数据目录
│   └── arknights_real_kb.txt
├── chroma_db/             # ChromaDB 向量数据库持久化目录
├── memory_db/             # 记忆系统持久化目录（SQLite）
└── memory/                # 记忆系统模块
    ├── __init__.py
    ├── base.py            # 核心数据结构：MemoryItem, MemoryConfig, BaseMemory
    ├── manager.py         # MemoryManager：统一协调调度
    ├── memory_tool.py     # MemoryTool：统一对外接口
    ├── embedding.py       # 嵌入服务
    ├── types/             # 四种记忆类型实现
    │   ├── working.py     # 工作记忆（内存，TTL）
    │   ├── episodic.py    # 情景记忆（SQLite）
    │   ├── semantic.py    # 语义记忆（Neo4j，可选）
    │   └── perceptual.py  # 感知记忆（Qdrant，可选）
    └── storage/           # 存储后端实现
        ├── document_store.py   # 文档存储（SQLite）
        ├── neo4j_store.py      # Neo4j 图存储
        └── qdrant_store.py     # Qdrant 向量存储
```

## 快速开始

### 环境要求

- Python 3.13+
- 有效的 DashScope API Key

### 安装依赖

```bash
pip install langchain langchain-chroma langchain-community streamlit \
            jieba python-dotenv dashscope
# 可选：记忆系统扩展存储
pip install neo4j qdrant-client
```

### 配置

1. 编辑 `.env` 文件，填入你的 `DASHSCOPE_API_KEY`
2. 可选：配置 Qdrant 和 Neo4j 云服务地址（用于语义记忆和感知记忆）
3. `config_data.py` 中可调整检索参数、模型参数、Prompt 模板等

### 启动

```bash
# 启动问答系统
streamlit run app_qa.py

# 启动知识库上传界面
streamlit run app_file_upload.py
```

访问浏览器显示的本地地址（通常是 `http://localhost:8501`）即可使用。

## 使用说明

### 问答系统 (app_qa.py)

1. 启动应用后，在聊天框中输入你的问题
2. 系统自动进行问题改写（处理指代消解），然后执行混合检索
3. 结合检索内容和对话历史，生成准确回答
4. 支持连续对话，系统会通过记忆系统持久化上下文

### 知识库更新 (app_file_upload.py)

1. 准备 TXT 格式的知识库文件
2. 通过文件上传页面上传文件
3. 系统自动进行语义文本分割（大文本）并向量化存储
4. 使用 MD5 校验避免重复上传

## 配置说明

在 `config_data.py` 中可调整以下参数：

### 向量数据库配置

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `collection_name` | Chroma 集合名称 | `rag` |
| `persist_directory` | 向量数据库持久化路径 | `./chroma_db` |

### 文本分割配置

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `chunk_size` | 文本块大小 | `1000` |
| `chunk_overlap` | 文本块重叠长度 | `100` |
| `max_split_len` | 触发语义分割的最小文本长度 | `200` |

### 检索配置

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `similarity_threshold` | 语义相似度阈值 | `5` |
| `top_k_semantic` | 语义检索返回文档数 | `3` |
| `top_k_keyword` | 关键词检索返回文档数 | `3` |
| `top_k_final` | 混合检索最终返回文档数 | `5` |
| `semantic_weight` | 语义相似度权重 | `0.6` |
| `keyword_weight` | 关键词权重 | `0.4` |

### 模型配置

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `embeddings_model_name` | 嵌入模型名称 | `text-embedding-v4` |
| `chat_model_name` | 对话模型名称 | `qwen3-max` |

### Prompt 模板

- `system_prompt` — 系统提示词
- `user_prompt` — 用户提示词
- `condense_question_system_template` — 问题改写提示词（处理指代消解）

## 知识库格式建议

知识库文件建议使用清晰的文本格式，例如：

```
【干员名称】XXX 【职业】XXX 【技能介绍】...
【关卡攻略】XXX 【推荐阵容】...
【游戏术语】XXX 【含义解释】...
```

## 开发说明

### 核心类说明

| 类名 | 说明 |
|------|------|
| `RagService` | RAG 服务主类，构建检索增强生成链（含问题改写层） |
| `KnowledgeBaseService` | 知识库管理服务，处理文本上传和向量化 |
| `VectorStoreService` | 向量存储服务封装，管理 ChromaDB + BM25 索引 |
| `BM25Retriever` | BM25 关键词检索器，基于 jieba 中文分词 |
| `HybridRetriever` | 混合检索器，融合语义相似度和关键词匹配 |
| `SemanticTextSplitter` | 语义文本分割器，支持智能重叠 |
| `MemoryTool` | 记忆系统统一接口（add/search/forget/consolidate） |
| `MemoryManager` | 记忆管理器，协调四种记忆类型的调度 |
| `FileChatMessageHistory` | 聊天历史管理，桥接 LangChain 与记忆系统 |

### 扩展方向

- 支持更多文件格式（PDF、Word 等）
- 添加图片识别功能
- 集成更多游戏数据源
- 优化检索算法和排序策略

## 注意事项

1. **API 费用**：使用 DashScope API 会产生费用，请注意控制用量
2. **数据安全**：不要将 `.env` 文件提交到公开仓库
3. **知识库质量**：回答质量依赖于知识库内容的准确性和完整性
4. **首次启动**：首次使用时需要加载向量数据库，可能需要一些时间
5. **记忆扩展**：语义记忆和感知记忆为可选项，需配置 Neo4j/Qdrant 云服务后才启用
