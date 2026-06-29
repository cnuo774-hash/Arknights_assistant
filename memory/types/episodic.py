# -*- coding: utf-8 -*-
"""
情景记忆（Episodic Memory）
SQLite + Qdrant 混合存储
评分公式: (向量相似度 x 0.8 + 时间近因性 x 0.2) x (0.8 + 重要性 x 0.4)
"""
import math
import time
from datetime import datetime
from typing import Optional
from ..base import BaseMemory, MemoryItem, MemorySearchResult, MemoryConfig, MemoryType
from ..embedding import EmbeddingService
from ..storage import QdrantStore, DocumentStore


class EpisodicMemory(BaseMemory):
    """情景记忆：SQLite + Qdrant"""

    def __init__(self, config: MemoryConfig):
        super().__init__(config)
        self._doc_store = DocumentStore(
            db_path=config.episodic_db_path,
            table_name="episodic_memories"
        )
        self._qdrant = None
        if QdrantStore and config.qdrant_url:
            self._qdrant = QdrantStore(
                url=config.qdrant_url,
                api_key=config.qdrant_api_key,
                collection_name=f"{config.qdrant_collection_prefix}_episodic",
                vector_size=config.qdrant_vector_size
            )
        self._embedding = EmbeddingService(
            provider=config.embedding_provider,
            vector_size=config.qdrant_vector_size
        )

    def add(self, item: MemoryItem) -> str:
        item.memory_type = MemoryType.EPISODIC
        self._doc_store.insert(item.to_dict())
        if not self._qdrant:
            return item.id
        vec = self._embedding.embed(item.content)
        self._qdrant.upsert(
            point_id=item.id,
            vector=vec,
            payload={
                "content": item.content,
                "session_id": item.session_id,
                "importance": item.importance,
                "timestamp": item.timestamp,
            }
        )
        return item.id

    def search(self, query: str, limit: int = 5, min_importance: float = 0.1,
               session_id: Optional[str] = None) -> list[MemorySearchResult]:
        filter_cond = {}
        if session_id:
            filter_cond["session_id"] = session_id
        if not self._qdrant:
            rows = self._doc_store.query(
                session_id=session_id,
                memory_type="episodic",
                min_importance=min_importance,
                order_by="timestamp DESC",
                limit=limit,
            )
            return [
                MemorySearchResult(
                    item=MemoryItem.from_dict(row),
                    score=self._calculate_recency_score(row.get("timestamp", "")),
                )
                for row in rows
            ]

        query_vec = self._embedding.embed(query)
        qdrant_results = self._qdrant.search(
            query_vector=query_vec,
            limit=limit * 2,
            filter_conditions=filter_cond if filter_cond else None
        )

        results = []
        for scored in qdrant_results:
            item_dict = self._doc_store.get_by_id(scored.id)
            if not item_dict:
                continue
            item = MemoryItem.from_dict(item_dict)
            if item.importance < min_importance:
                continue

            vec_sim = scored.score
            recency = self._calculate_recency_score(item.timestamp)
            score = (vec_sim * 0.8 + recency * 0.2) * (0.8 + item.importance * 0.4)
            results.append(MemorySearchResult(item=item, score=score))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    def _calculate_recency_score(self, timestamp: str) -> float:
        """时间近因性得分（指数衰减）"""
        try:
            memory_time = datetime.fromisoformat(timestamp)
            age_hours = (datetime.now() - memory_time).total_seconds() / 3600
            decay = math.exp(-0.1 * age_hours / 24)
            return max(0.1, decay)
        except Exception:
            return 0.5

    def forget(self, strategy: str = "importance", threshold: float = 0.3,
               older_than_hours: Optional[int] = None,
               capacity_ratio: float = 0.9) -> int:
        removed = 0
        if strategy == "importance":
            to_delete = self._doc_store.query(
                memory_type="episodic", min_importance=0.0, limit=10000
            )
            ids = [d["id"] for d in to_delete if d["importance"] < threshold]
            if ids:
                if self._qdrant:
                    self._qdrant.delete(ids)
                removed = self._doc_store.delete_by_ids(ids)
        elif strategy == "time" and older_than_hours:
            to_delete = self._doc_store.query_older_than(
                hours=older_than_hours, memory_type="episodic"
            )
            ids = [d["id"] for d in to_delete]
            if ids:
                if self._qdrant:
                    self._qdrant.delete(ids)
                removed = self._doc_store.delete_by_ids(ids)
        elif strategy == "capacity":
            total = self._doc_store.count(memory_type="episodic")
            limit_count = int(total * capacity_ratio)
            excess = total - limit_count
            if excess > 0:
                lowest = self._doc_store.get_lowest_importance(
                    limit=excess, memory_type="episodic"
                )
                ids = [d["id"] for d in lowest]
                if ids:
                    if self._qdrant:
                        self._qdrant.delete(ids)
                    removed = self._doc_store.delete_by_ids(ids)
        return removed

    def count(self) -> int:
        return self._doc_store.count(memory_type="episodic")

    def clear(self) -> None:
        self._doc_store.clear(memory_type="episodic")
        if self._qdrant:
            self._qdrant.clear()
