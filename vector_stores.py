from langchain_chroma import Chroma
import config_data as config
from langchain_core.documents import Document
from typing import List
import jieba
import math
from collections import Counter
import hashlib

class BM25Retriever:
    """基于BM25算法的关键词检索器"""
    
    def __init__(self, documents: List[Document], k1=1.5, b=0.75):
        self.k1 = k1  # BM25参数k1
        self.b = b    # BM25参数b
        self.documents = documents
        
        # 构建倒排索引
        self.doc_freq = Counter()  # 包含每个词的文档数
        self.term_freq = []  # 每个文档的词频
        self.doc_lengths = []  # 每个文档的长度
        self.avg_doc_length = 0
        
        self._build_index()
    
    def _tokenize(self, text: str) -> List[str]:
        """中文分词"""
        # 使用jieba进行中文分词
        words = jieba.lcut(text.lower())
        # 过滤停用词和单字符
        stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', 
                     '不', '人', '都', '一', '一个', '上', '也', '很', 
                     '到', '说', '要', '去', '你', '会', '着', '没有', 
                     '看', '好', '自己', '这', '他', '她', '它', '们'}
        return [w for w in words if len(w) > 1 and w not in stop_words]
    
    def _build_index(self):
        """构建BM25索引"""
        n_docs = len(self.documents)
        
        for doc in self.documents:
            tokens = self._tokenize(doc.page_content)
            self.doc_lengths.append(len(tokens))
            
            # 计算词频
            tf = Counter(tokens)
            self.term_freq.append(tf)
            
            # 更新文档频率
            for term in set(tokens):
                self.doc_freq[term] += 1
        
        # 计算平均文档长度
        self.avg_doc_length = sum(self.doc_lengths) / n_docs if n_docs > 0 else 0
    
    def _bm25_score(self, query_tokens: List[str], doc_idx: int) -> float:
        """计算查询和文档的BM25分数"""
        score = 0.0
        tf = self.term_freq[doc_idx]
        doc_len = self.doc_lengths[doc_idx]
        n_docs = len(self.documents)
        
        for token in query_tokens:
            if token in tf:
                # 词频
                freq = tf[token]
                # 文档频率
                doc_freq = self.doc_freq.get(token, 0)
                
                # IDF (逆文档频率)
                idf = math.log((n_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)
                
                # TF (词频归一化)
                tf_norm = (freq * (self.k1 + 1)) / (freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length))
                
                score += idf * tf_norm
        
        return score
    
    def get_relevant_documents(self, query: str, k: int = 5) -> List[Document]:
        """获取与查询相关的文档"""
        query_tokens = self._tokenize(query)
        
        if not query_tokens:
            return []
        
        # 计算所有文档的BM25分数
        scores = []
        for idx in range(len(self.documents)):
            score = self._bm25_score(query_tokens, idx)
            if score > 0:
                scores.append((idx, score))
        
        # 按分数排序
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # 返回前k个文档
        result_docs = []
        for idx, score in scores[:k]:
            doc = self.documents[idx].copy()
            doc.metadata['bm25_score'] = score
            doc.metadata['retrieval_method'] = 'keyword'
            result_docs.append(doc)
        
        return result_docs


class HybridRetriever:
    """混合检索器：结合语义相似度和关键词匹配"""
    
    def __init__(self, vector_store, bm25_retriever: BM25Retriever = None):
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
    
    def invoke(self, query: str) -> List[Document]:
        """执行混合检索"""
        # 获取配置参数
        top_k_semantic = getattr(config, 'top_k_semantic', 3)
        top_k_keyword = getattr(config, 'top_k_keyword', 3)
        top_k_final = getattr(config, 'top_k_final', 5)
        semantic_weight = getattr(config, 'semantic_weight', 0.6)
        keyword_weight = getattr(config, 'keyword_weight', 0.4)
        
        # 1. 语义检索
        semantic_docs = self.vector_store.similarity_search(
            query, 
            k=top_k_semantic
        )
        
        # 为语义检索结果添加分数（基于相似度距离转换）
        for doc in semantic_docs:
            if 'distance' in doc.metadata:
                # Chroma返回的距离，转换为相似度分数
                doc.metadata['semantic_score'] = 1.0 / (1.0 + doc.metadata.get('distance', 1.0))
            else:
                doc.metadata['semantic_score'] = 1.0
            doc.metadata['retrieval_method'] = doc.metadata.get('retrieval_method', 'semantic')
        
        # 2. 关键词检索
        keyword_docs = []
        if self.bm25_retriever:
            keyword_docs = self.bm25_retriever.get_relevant_documents(
                query, 
                k=top_k_keyword
            )
        
        # 3. 合并结果并去重
        all_docs = self._merge_and_rerank(
            semantic_docs, 
            keyword_docs,
            semantic_weight,
            keyword_weight,
            top_k_final
        )
        
        return all_docs

    def _merge_and_rerank(self, semantic_docs, keyword_docs, semantic_weight, keyword_weight, top_k):
        doc_dict = {}

        # 辅助函数：提取归一化分数
        def normalize_scores(docs, score_key):
            if not docs: return
            max_score = max((doc.metadata.get(score_key, 0.0) for doc in docs), default=0.0)
            min_score = min((doc.metadata.get(score_key, 0.0) for doc in docs), default=0.0)

            for doc in docs:
                raw_score = doc.metadata.get(score_key, 0.0)
                if max_score == min_score:
                    doc.metadata[f'{score_key}_norm'] = 1.0 if max_score > 0 else 0.0
                else:
                    doc.metadata[f'{score_key}_norm'] = (raw_score - min_score) / (max_score - min_score)

        normalize_scores(semantic_docs, 'semantic_score')
        normalize_scores(keyword_docs, 'bm25_score')

        # 合并逻辑，使用 MD5 去重
        for doc in semantic_docs:
            doc_key = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
            score = doc.metadata.get('semantic_score_norm', 0.0)
            doc_dict[doc_key] = {
                'doc': doc,
                'semantic_score_norm': score,  # 修复：补全缺失的键值
                'keyword_score_norm': 0.0,
                'final_score': score * semantic_weight
            }

        for doc in keyword_docs:
            doc_key = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
            score = doc.metadata.get('bm25_score_norm', 0.0)
            if doc_key in doc_dict:
                # 命中交集，补充关键词分数
                doc_dict[doc_key]['keyword_score_norm'] = score
                doc_dict[doc_key]['final_score'] += score * keyword_weight
                doc_dict[doc_key]['doc'].metadata['retrieval_method'] = 'hybrid'
            else:
                # 仅在关键词检索中命中
                doc_dict[doc_key] = {
                    'doc': doc,
                    'semantic_score_norm': 0.0,
                    'keyword_score_norm': score,
                    'final_score': score * keyword_weight
                }

        # 按最终分数排序
        sorted_items = sorted(
            doc_dict.values(),
            key=lambda x: x['final_score'],
            reverse=True
        )

        # 返回前k个文档
        result_docs = []
        for item in sorted_items[:top_k]:
            doc = item['doc']
            # 修复：正确读取归一化后的分数，避免 KeyError
            doc.metadata['final_score'] = item['final_score']
            doc.metadata['semantic_score_norm'] = item['semantic_score_norm']
            doc.metadata['keyword_score_norm'] = item['keyword_score_norm']
            result_docs.append(doc)

        return result_docs


class VectorStoreService:
    def __init__(self, embedding):
        self.embedding = embedding
        self.vector_store = Chroma(
            collection_name=config.collection_name,
            embedding_function=self.embedding,
            persist_directory=config.persist_directory
        )
        
        # 初始化BM25检索器
        self.bm25_retriever = None
        self._initialize_bm25()
    
    def _initialize_bm25(self):
        """从向量数据库中获取所有文档来初始化BM25检索器"""
        try:
            # 获取所有文档
            all_docs = self.vector_store.get()
            
            if all_docs and all_docs['documents']:
                documents = [
                    Document(
                        page_content=doc,
                        metadata=all_docs['metadatas'][i]
                    )
                    for i, doc in enumerate(all_docs['documents'])
                ]
                
                self.bm25_retriever = BM25Retriever(documents)
        except Exception as e:
            print(f"初始化BM25检索器失败: {e}")
            self.bm25_retriever = None
    
    def update_bm25_index(self):
        """更新BM25索引（在添加新文档后调用）"""
        self._initialize_bm25()
    
    def get_retriever(self):
        """获取混合检索器"""
        return HybridRetriever(self.vector_store, self.bm25_retriever)
    
    def add_documents_with_bm25_update(self, texts: list, metadatas: list = None):
        """添加文档并更新BM25索引"""
        self.vector_store.add_texts(texts, metadatas)
        # 添加文档后更新BM25索引
        self.update_bm25_index()
