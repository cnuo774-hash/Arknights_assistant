# -*- coding: utf-8 -*-
"""
统一嵌入服务
支持 DashScope（阿里云）、本地 TF-IDF 回退
"""
import os
import math
from collections import Counter
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class EmbeddingService:
    """统一嵌入服务，优先使用 DashScope，失败时回退到 TF-IDF"""

    def __init__(self, provider: str = "dashscope", vector_size: int = 1024):
        self.provider = provider
        self.vector_size = vector_size
        self._dashscope_embeddings = None
        self._tfidf_vocab: dict[str, int] = {}
        self._tfidf_idf: dict[str, float] = {}
        self._tfidf_doc_count = 0

        if provider == "dashscope":
            try:
                from langchain_community.embeddings import DashScopeEmbeddings
                self._dashscope_embeddings = DashScopeEmbeddings(
                    model="text-embedding-v4",
                    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
                )
            except Exception:
                self._dashscope_embeddings = None

    def embed(self, text: str) -> list[float]:
        """对单段文本进行向量化"""
        if self._dashscope_embeddings:
            try:
                result = self._dashscope_embeddings.embed_query(text)
                if result and len(result) == self.vector_size:
                    return list(result)
                if result:
                    return list(result)
            except Exception:
                pass
        return self._tfidf_embed(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量向量化"""
        if self._dashscope_embeddings:
            try:
                results = self._dashscope_embeddings.embed_documents(texts)
                return [list(r) for r in results]
            except Exception:
                pass
        return [self._tfidf_embed(t) for t in texts]

    # ---------- TF-IDF ----------

    def _tokenize(self, text: str) -> list[str]:
        import re
        tokens = re.findall(r'[\u4e00-\u9fff]|[a-zA-Z0-9]+', text.lower())
        return tokens

    def build_tfidf_vocab(self, corpus: list[str]):
        """基于语料库构建 TF-IDF 词汇表"""
        self._tfidf_doc_count = len(corpus)
        df = Counter()
        for doc in corpus:
            tokens = set(self._tokenize(doc))
            for token in tokens:
                df[token] += 1
        self._tfidf_idf = {
            token: math.log((self._tfidf_doc_count + 1) / (freq + 1)) + 1
            for token, freq in df.items()
        }
        sorted_tokens = sorted(self._tfidf_idf.keys(),
                               key=lambda t: self._tfidf_idf[t], reverse=True)
        self._tfidf_vocab = {t: i for i, t in enumerate(sorted_tokens[:self.vector_size])}

    def _tfidf_embed(self, text: str) -> list[float]:
        """使用 TF-IDF 向量化"""
        tokens = self._tokenize(text)
        tf = Counter(tokens)
        vec = [0.0] * self.vector_size
        for token, freq in tf.items():
            if token in self._tfidf_vocab:
                idx = self._tfidf_vocab[token]
                tf_val = freq / max(len(tokens), 1)
                idf_val = self._tfidf_idf.get(token, 1.0)
                vec[idx] = tf_val * idf_val
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """计算余弦相似度"""
        if not a or not b:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
