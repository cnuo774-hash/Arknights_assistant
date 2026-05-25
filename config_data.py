#项目配置文件

md5_path="./md5_txt"

#Chroma
collection_name="rag"
persist_directory="./chroma_db"

#Splitter
chunk_size=1000
chunk_overlap=100
max_split_len=200

# 检索配置
similarity_threshold = 5
top_k_semantic = 3  # 语义检索返回的文档数
top_k_keyword = 3   # 关键词检索返回的文档数
top_k_final = 5     # 混合检索后最终返回的文档数

# 混合检索权重配置
semantic_weight = 0.6  # 语义相似度权重
keyword_weight = 0.4   # 关键词权重

#modle
embeddings_model_name="text-embedding-v4"
chat_model_name="qwen3-max"

#prompt
system_prompt="""
你是网络游戏明日方舟的专业攻略助手，以我提供的参考资料为主，简洁明了的回答用户的问题，参考资料{context}
。此外，
1.你可以有自己的观点，但是不能生成不存在的资料
2.所有资料都基于我提供的知识库和你模型中自带的知识库，每个资料都要在库中找到对于依据
,并且我提示用户的历史对话记录，如下：
"""
user_prompt="""
请回答用户提问，{input}
"""

session_config={
    "configurable":{
        "session_id":"user_001",
    }
}

condense_question_system_template = """
        给定以下对话历史和用户最新的问题，该问题可能引用了历史对话中的上下文（比如使用了“他”、“这个”等代词）。
        请将这个问题重写为一个独立的、完整的、自包含的问题，使其在没有历史记录的情况下也能被完全理解。
        注意：
        1. 你的任务仅仅是重写问题，**不要**回答问题。
        2. 如果问题本身已经很独立，不需要重写，请直接原样返回。
        """