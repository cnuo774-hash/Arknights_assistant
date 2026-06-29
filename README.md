# 明日方舟攻略助手

基于 RAG、混合检索和多类型记忆系统的本地 Web 问答应用，用于查询《明日方舟》干员养成、关卡打法、机制说明和知识库资料。

## 功能

- 流式问答：FastAPI SSE 实时返回模型输出。
- 混合检索：Chroma 语义检索 + BM25 关键词检索加权融合。
- 知识库上传：支持 UTF-8 TXT 文件上传、MD5 去重、语义分块和向量化入库。
- 多轮对话：通过记忆系统保存上下文，支持连续追问。
- 单一前端：`static/index.html` 是无构建静态页面，不再保留 Streamlit 方案。

## 技术栈

| 模块 | 技术 |
| --- | --- |
| 前端 | HTML + CSS + 原生 JavaScript |
| 后端 | FastAPI + Uvicorn |
| RAG 编排 | LangChain |
| 对话模型 | 通义千问 `qwen3-max` |
| 嵌入模型 | DashScope `text-embedding-v4` |
| 向量库 | ChromaDB |
| 关键词检索 | jieba + BM25 |
| 记忆存储 | SQLite，Neo4j/Qdrant 可扩展 |

## 项目结构

```text
Arknights_assistant/
├── api.py                  # FastAPI 接口与静态前端入口
├── static/
│   └── index.html          # Web 前端
├── scripts/
│   └── start_web.bat       # Windows 启动脚本
├── run_web.bat             # 兼容入口，转发到 scripts/start_web.bat
├── requirements_web.txt    # Web 运行依赖
├── rag.py                  # RAG 链与问题改写
├── knowledge_base.py       # 知识库上传、分块、入库
├── vector_stores.py        # Chroma + BM25 混合检索
├── file_history_store.py   # LangChain 历史记录适配层
├── config_data.py          # 模型、Prompt、检索参数
├── memory/                 # 记忆系统模块
├── data/                   # 原始知识库资料
├── chroma_db/              # Chroma 持久化目录
├── memory_db/              # SQLite 记忆库
├── md5_txt                 # 上传去重记录
└── .gitignore              # 忽略密钥、缓存和运行数据
```

## 环境要求

- Python 3.11+
- 可用的 DashScope API Key

## 安装

```bash
pip install -r requirements_web.txt
```

创建或编辑 `.env`：

```env
DASHSCOPE_API_KEY=your-api-key-here
```

## 启动

Windows 可以双击 `run_web.bat`，也可以运行正式脚本：

```bat
scripts\start_web.bat
```

手动启动：

```bash
uvicorn api:app --host 0.0.0.0 --port 8090 --reload
```

浏览器打开：

```text
http://localhost:8090
```

## 使用

1. 在聊天输入框提问，例如“山推荐专精哪个技能？”。
2. 需要更新资料时，在右侧知识库面板上传 `.txt` 文件。
3. 上传成功后，下一次提问会自动重建 RAG 服务并使用新检索索引。
4. “清除当前对话”会清理当前 session 的持久化历史记录。

## API

| 路径 | 方法 | 说明 |
| --- | --- | --- |
| `/` | GET | 返回前端页面 |
| `/api/chat` | POST | SSE 流式问答 |
| `/api/knowledge/upload` | POST | 上传 TXT 知识文件 |
| `/api/history/{session_id}` | GET | 获取会话历史 |
| `/api/history/{session_id}` | DELETE | 清除会话历史 |

聊天请求示例：

```json
{
  "prompt": "山推荐专精哪个技能？",
  "session_id": "user_001"
}
```

## 配置

主要参数在 `config_data.py` 中调整：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `chat_model_name` | `qwen3-max` | 对话模型 |
| `embeddings_model_name` | `text-embedding-v4` | 嵌入模型 |
| `chunk_size` | `1000` | 分块大小 |
| `chunk_overlap` | `100` | 分块重叠 |
| `top_k_semantic` | `3` | 语义检索返回数 |
| `top_k_keyword` | `3` | 关键词检索返回数 |
| `top_k_final` | `5` | 最终上下文数量 |
| `semantic_weight` | `0.6` | 语义检索权重 |
| `keyword_weight` | `0.4` | 关键词检索权重 |

## 知识库文本建议

```text
【干员】山
【定位】近卫 / 单守一路
【技能】二技能适合常驻输出与自回复，三技能偏爆发控制。

【关卡】LS-6
【推荐】先锋回费，群攻清杂，单体术师处理高防单位。
```

## 注意事项

- DashScope 调用会产生费用，请控制调用量。
- 回答质量取决于知识库覆盖范围和文本质量。
- 首次启动或知识库更新后，RAG 服务初始化需要等待向量库和 BM25 索引加载。
- Neo4j 和外部 Qdrant 属于扩展能力，未配置时核心 Web 问答仍可运行。
