# 🎮 明日方舟攻略助手 (Arknights Assistant)

一个基于 RAG（Retrieval-Augmented Generation）技术的智能问答系统，专为《明日方舟》玩家提供准确、专业的游戏攻略和知识查询服务。

## ✨ 功能特性

- 🤖 **智能问答**：基于大语言模型的智能对话，提供准确的游戏攻略
- 📚 **知识库管理**：支持自定义知识库的上传和更新
- 🔍 **语义检索**：使用向量数据库实现高效的语义搜索
- 💬 **对话历史**：自动维护对话上下文，提供更连贯的交互体验
- 🌐 **Web 界面**：基于 Streamlit 的友好用户界面
- 📄 **文件上传**：支持 TXT 格式的知识库文件上传

## 🛠️ 技术栈

### 核心框架

- **LangChain** - LLM 应用开发框架
- **Streamlit** - Web 界面构建
- **ChromaDB** - 向量数据库

### AI 模型

- **通义千问 (Qwen3-Max)** - 对话生成模型
- **DashScope Embeddings (text-embedding-v4)** - 文本嵌入模型

### 其他依赖

- Python 3.x
- dotenv - 环境变量管理
- hashlib - MD5 去重校验

## 📁 项目结构

Arknights_assistant/ 

├── app_qa.py # 问答系统主应用 

├── app_file_upload.py # 知识库上传应用

├── rag.py # RAG 服务核心逻辑 

├── knowledge_base.py # 知识库管理服务

 ├── vector_stores.py # 向量存储封装 

├── config_data.py # 项目配置文件 

├── .env # 环境变量配置

 ├── data/ # 知识库数据目录 

├── chroma_db/ # 向量数据库持久化目录 

├── chat_history/ # 聊天记录存储 

└── md5_txt # MD5 去重记录



访问浏览器显示的本地地址（通常是 `http://localhost:8501`）即可使用。

## 📖 使用说明

### 问答系统

1. 启动应用后，在聊天框中输入你的问题
2. 系统会自动从知识库中检索相关信息
3. 结合检索内容和对话历史，生成准确的回答
4. 支持连续对话，系统会记住之前的对话内容

### 知识库更新

1. 准备 TXT 格式的知识库文件
2. 通过文件上传页面上传文件
3. 系统会自动进行文本分割、向量化并存储
4. 使用 MD5 校验避免重复上传

## ⚙️ 配置说明

在 `config_data.py` 中可以调整以下参数：

### 向量数据库配置

- `collection_name`: Chroma 集合名称
- `persist_directory`: 向量数据库持久化路径

### 文本分割配置

- `chunk_size`: 文本块大小（默认 1000）
- `chunk_overlap`: 文本块重叠长度（默认 100）
- `separators`: 分割分隔符列表
- `max_split_len`: 最大分割长度阈值

### 模型配置

- `embeddings_model_name`: 嵌入模型名称
- `chat_model_name`: 对话模型名称

### Prompt 模板

- `system_prompt`: 系统提示词
- `user_prompt`: 用户提示词

## 📝 知识库格式建议

知识库文件建议使用清晰的文本格式，例如：

【干员名称】XXX 【职业】XXX 【技能介绍】...
【关卡攻略】XXX 【推荐阵容】...
【游戏术语】XXX 【含义解释】...

## 🔧 开发说明

### 核心类说明

- **RagService**: RAG 服务主类，负责构建检索增强生成链
- **KnowledgeBaseService**: 知识库管理服务，处理文本上传和向量化
- **VectorStoreService**: 向量存储服务封装

### 扩展功能

可以在此基础上扩展：

- 支持更多文件格式（PDF、Word 等）
- 添加图片识别功能
- 集成更多游戏数据源
- 优化检索算法和排序策略

## ⚠️ 注意事项

1. **API 费用**：使用 DashScope API 会产生费用，请注意控制用量
2. **数据安全**：不要将 `.env` 文件提交到公开仓库
3. **知识库质量**：回答质量依赖于知识库内容的准确性和完整性
4. **首次启动**：首次使用时需要加载向量数据库，可能需要一些时间



---

