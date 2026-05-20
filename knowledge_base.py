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
    """基于语义的智能文本分割器"""

    def __init__(self, chunk_size=1000, chunk_overlap=100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_by_sentences(self, text: str) -> list:
        """将文本按句子分割（准确保留中英文标点）"""
        sentence_pattern = r'(?<=[。！？.!?\n])'
        sentences = re.split(sentence_pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def _get_boundary_indices(self, text: str) -> list:
        """
        同时兼容中文（单字/标点）与英文（完整单词），不破坏原文本空格。
        """
        # 匹配中文字符、英文单词/数字连体、或者任意非空白标点
        pattern = r'[\u4e00-\u9fff]|[a-zA-Z0-9_\-]+|[^\s\w]'
        return [(m.start(), m.end()) for m in re.finditer(pattern, text)]

    def _cosine_similarity(self, vec1: list, vec2: list) -> float:
        """计算两个向量的余弦相似度"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    def smart_overlap(self, chunks: list, embeddings, overlap_ratio=0.1) -> list:
        """智能重叠：根据语义相似度动态调整重叠内容"""
        if len(chunks) <= 1:
            return chunks

        enhanced_chunks = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlap_text = self._get_smart_overlap(chunk, chunks[i+1], embeddings, overlap_ratio)
                enhanced_chunks.append(chunk + overlap_text)
            elif i == len(chunks) - 1:
                overlap_text = self._get_smart_overlap(chunks[i-1], chunk, embeddings, overlap_ratio)
                enhanced_chunks.append(overlap_text + chunk)
            else:
                prev_overlap = self._get_smart_overlap(chunks[i-1], chunk, embeddings, overlap_ratio)
                next_overlap = self._get_smart_overlap(chunk, chunks[i+1], embeddings, overlap_ratio)
                enhanced_chunks.append(prev_overlap + chunk + next_overlap)

        return enhanced_chunks

    def _get_smart_overlap(self, prev_chunk: str, curr_chunk: str, embeddings, overlap_ratio: float) -> str:
        """获取智能重叠文本（优化版：中英兼容 + 批量映射 + 稀释采样）"""
        # 1. 获取前一个分块的单元边界
        prev_boundaries = self._get_boundary_indices(prev_chunk)
        total_prev_units = len(prev_boundaries)
        overlap_units = int(total_prev_units * overlap_ratio)

        if overlap_units == 0:
            return ""

        # 锁定右侧（当前分块）的“固定比较锚点”
        # 固定取当前分块开头的若干个单元作为对比参照物，避免在循环中不断改变目标
        curr_boundaries = self._get_boundary_indices(curr_chunk)
        anchor_units_count = min(len(curr_boundaries), overlap_units * 2, 40)
        if anchor_units_count == 0:
            return ""
        anchor_end_pos = curr_boundaries[anchor_units_count - 1][1]
        anchor_text = curr_chunk[:anchor_end_pos]

        #构造左侧（前一个分块末尾）的候选重叠文本
        #稀释采样。没必要逐字对比，限制最大候选数量（如最多8个窗口），大幅降低计算量
        max_candidates = 8
        step = max(1, overlap_units // max_candidates)
        candidate_lengths = list(range(overlap_units, 0, -step))

        candidates = []
        for length in candidate_lengths:
            start_pos = prev_boundaries[-length][0]
            candidates.append(prev_chunk[start_pos:])

        if not candidates:
            return ""

        #利用批量接口减少网络 IO 开销（性能暴涨的关键）
        try:
            candidate_embs = embeddings.embed_documents(candidates)
            anchor_emb = embeddings.embed_query(anchor_text)
        except AttributeError:
            anchor_emb = embeddings.embed_query(anchor_text)
            candidate_embs = [embeddings.embed_query(c) for c in candidates]

        best_overlap = ""
        best_similarity = 0.3  # 语义相似度门槛

        for candidate_text, cand_emb in zip(candidates, candidate_embs):
            similarity = self._cosine_similarity(cand_emb, anchor_emb)
            if similarity > best_similarity:
                best_similarity = similarity
                best_overlap = candidate_text

        return best_overlap

    def semantic_split(self, text: str, embeddings=None) -> list:
        """基于语义的文本分割主函数"""
        sentences = self.split_by_sentences(text)
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

