"""
知识库的基础服务
"""
import os
import config_data as config
import hashlib
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from datetime import datetime
from dotenv import load_dotenv
import re
import jieba

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


class SemanticTextSplitter:
    """基于语义的文本分割器"""
    
    def __init__(self, chunk_size=1000, chunk_overlap=100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def split_by_sentences(self, text: str) -> list:
        """将文本按句子分割"""
        # 匹配中英文句子结束符
        sentence_pattern = r'(?<=[。！？.!?\n])'
        sentences = re.split(sentence_pattern, text)
        # 过滤空字符串并去除空白
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def calculate_semantic_similarity(self, text1: str, text2: str, embeddings) -> float:
        """计算两段文本的语义相似度"""
        try:
            emb1 = embeddings.embed_query(text1)
            emb2 = embeddings.embed_query(text2)

            dot_product = sum(a * b for a, b in zip(emb1, emb2))
            norm1 = sum(a * a for a in emb1) ** 0.5
            norm2 = sum(b * b for b in emb2) ** 0.5
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return similarity
        except:
            return 0.0
    
    def smart_overlap(self, chunks: list, embeddings, overlap_ratio=0.1) -> list:
        """智能重叠：根据语义相似度动态调整重叠内容"""
        if len(chunks) <= 1:
            return chunks
        
        enhanced_chunks = []
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                # 第一个chunk，只添加后续重叠
                if len(chunks) > 1:
                    overlap_text = self._get_smart_overlap(chunk, chunks[i+1], embeddings, overlap_ratio)
                    enhanced_chunks.append(chunk + overlap_text)
                else:
                    enhanced_chunks.append(chunk)
            elif i == len(chunks) - 1:
                # 最后一个chunk，只添加前序重叠
                overlap_text = self._get_smart_overlap(chunks[i-1], chunk, embeddings, overlap_ratio)
                enhanced_chunks.append(overlap_text + chunk)
            else:
                # 中间chunk，添加前后重叠
                prev_overlap = self._get_smart_overlap(chunks[i-1], chunk, embeddings, overlap_ratio)
                next_overlap = self._get_smart_overlap(chunk, chunks[i+1], embeddings, overlap_ratio)
                enhanced_chunks.append(prev_overlap + chunk + next_overlap)
        
        return enhanced_chunks
    
    def _get_smart_overlap(self, prev_chunk: str, curr_chunk: str, embeddings, overlap_ratio: float) -> str:
        """获取智能重叠文本"""
        # 从前一个chunk的末尾提取可能的重叠部分
        words = prev_chunk.split()
        overlap_length = int(len(words) * overlap_ratio)
        
        if overlap_length == 0:
            return ""
        
        # 尝试不同长度的重叠，找到语义最相似的部分
        best_overlap = ""
        best_similarity = 0
        
        for length in range(min(overlap_length, len(words)), 0, -1):
            candidate = " ".join(words[-length:])
            similarity = self.calculate_semantic_similarity(candidate, curr_chunk[:len(candidate)*2], embeddings)
            
            if similarity > best_similarity and similarity > 0.3:  # 相似度阈值
                best_overlap = candidate
                best_similarity = similarity
        
        return best_overlap if best_overlap else ""
    
    def semantic_split(self, text: str, embeddings=None) -> list:
        """基于语义的文本分割"""
        # 第一步：按句子分割
        sentences = self.split_by_sentences(text)
        
        # 第二步：合并句子成合理大小的chunks
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)

        if embeddings and len(chunks) > 1:
            chunks = self.smart_overlap(chunks, embeddings)
        
        return chunks


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

        # 新增语义分割器
        self.semantic_spliter = SemanticTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )

    def upload_by_str(self,data:str,file):
        #将传入的字符串向量化，并保存到向量数据库中
        md5_hex=get_string_md5(data)
        if check_md5(md5_hex):
            return "[跳过]数据已存在于数据库中"
        
        # 使用语义分割
        if len(data)>config.max_split_len:
            # 获取嵌入函数用于智能重叠
            embeddings = self.chroma._embedding_function
            knowledge_chunks=self.semantic_spliter.semantic_split(data, embeddings)
        else:
            knowledge_chunks=[data]

        metadata={
            "source":file,
            "md5":md5_hex,
            "create_time":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        self.chroma.add_texts(
            knowledge_chunks,
            metadatas=[metadata for _ in knowledge_chunks]
        )

        save_md5(md5_hex)
        return "[成功]内容已上传向量库（使用语义分割）"

