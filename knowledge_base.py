"""
知识库的基础服务
"""
import os
import config_data as config
import hashlib
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime
from dotenv import load_dotenv

def check_md5(md5_str:str):
    if not os.path.exists(config.md5_path):
        open(config.md5_path,"w",encoding="utf-8").close()
        return False
    for line in open(config.md5_path,"r",encoding="utf-8").readlines():
        if line.strip()==md5_str:
            return True
    return False

def save_md5(md5_str:str):
    with open(config.md5_path,"a",encoding="utf-8") as f:
        f.write(md5_str+"\n")

def get_string_md5(input_str:str, encoding="utf-8"):
    str_bytes=input_str.encode(encoding=encoding)
    md5=hashlib.md5()
    md5.update(str_bytes)
    return md5.hexdigest()



class KnowledgeBaseService:
    def __init__(self):
        load_dotenv()

        os.makedirs(config.persist_directory,exist_ok=True)

        self.chroma=Chroma(
            collection_name=config.collection_name,
            embedding_function=DashScopeEmbeddings(
                model="text-embedding-v4",
                dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
            ),
            persist_directory=config.persist_directory
        ) #向量数据库的实例对象
        self.spliter=RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size, #文本段分割的最大长度
            chunk_overlap=config.chunk_overlap, #连续文本段允许的最大重复次数
            separators=config.separators, #文本段分割的分隔符
            length_function=len,
        ) #文本分割器的对象

    def upload_by_str(self,data:str,file):
        #将传入的字符串向量化，并保存到向量数据库中
        md5_hex=get_string_md5(data)
        if check_md5(md5_hex):
            return "[跳过]数据已存在于数据库中"
        if len(data)>config.max_split_len:
            knowledge_chunks=self.spliter.split_text(data)
        else:
            knowledge_chunks=[data]

        metadata={
            "source":file,
            "md5":md5_hex,
            "create_time":datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        self.chroma.add_texts(
            knowledge_chunks,
            metadatas=[metadata for _ in knowledge_chunks]
        )

        save_md5(md5_hex)
        return "[成功]内容已上传向量库"

