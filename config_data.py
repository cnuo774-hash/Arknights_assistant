#项目配置文件

md5_path="./md5_txt"

#Chroma
collection_name="rag"
persist_directory="./chroma_db"

#Splitter
chunk_size=1000
chunk_overlap=100
max_split_len=200

#
similarity_threshold = 5

#modle
embeddings_model_name="text-embedding-v4"
chat_model_name="qwen3-max"

#prompt
system_prompt="""
你是网络游戏明日方舟的专业攻略助手，以我提供的参考资料为主，简洁明了的回答用户的问题，参考资料{context},并且我提示用户的历史对话记录，如下：
"""
user_prompt="""
请回答用户提问，{input}
"""

session_config={
    "configurable":{
        "session_id":"user_001",
    }
}