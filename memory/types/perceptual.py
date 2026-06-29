# -*- coding: utf-8 -*-
"""
感知记忆（Perceptual Memory）
多模态支持：文本、图像、音频、视频
SQLite + Qdrant 混合存储
评分公式: (向量相似度 x 0.8 + 时间近因性 x 0.2) x (0.8 + 重要性 x 0.4)
"""
import math
from datetime import datetime
from typing import Optional
from ..base import BaseMemory, MemoryItem, MemorySearchResult, MemoryConfig, MemoryType, Modality
from ..embedding import EmbeddingService
from ..storage import QdrantStore, DocumentStore


class PerceptualMemory(BaseMemory):
    """感知记忆：多模态 SQLite + Qdrant"""

    def __init__(self, config: MemoryConfig):
        super().__init__(config)
        self._doc_store = DocumentStore(
            db_path=config.perceptual_db_path,
            table_name="perceptual_memories"
        )
        if not QdrantStore:
            raise RuntimeError("qdrant-client is required for perceptual memory")
        self._qdrant_stores: dict[Modality, QdrantStore] = {}
        for modality in [Modality.TEXT, Modality.IMAGE, Modality.AUDIO, Modality.VIDEO]:
            self._qdrant_stores[modality] = QdrantStore(
                url=config.qdrant_url,
                api_key=config.qdrant_api_key,
                collection_name=f"{config.qdrant_collection_prefix}_perceptual_{modality.value}",
                vector_size=config.qdrant_vector_size
            )
        self._embedding = EmbeddingService(
            provider=config.embedding_provider,
            vector_size=config.qdrant_vector_size
        )

    def add(self, item: MemoryItem) -> str:
        item.memory_type = MemoryType.PERCEPTUAL
        self._doc_store.insert(item.to_dict())
        vec = self._embedding.embed(item.content)
        store = self._qdrant_stores.get(item.modality, self._qdrant_stores[Modality.TEXT])
        store.upsert(
            point_id=item.id,
            vector=vec,
            payload={
                "content": item.content,
                "session_id": item.session_id,
                "importance": item.importance,
                "modality": item.modality.value,
                "timestamp": item.timestamp,
            }
        )
        return item.id

    def search(self, query: str, limit: int = 5, min_importance: float = 0.1,
               session_id: Optional[str] = None,
               modality: Optional[Modality] = None) -> list[MemorySearchResult]:
        query_vec = self._embedding.embed(query)

        stores_to_search: list[QdrantStore] = []
        if modality:
            stores_to_search = [self._qdrant_stores.get(modality,
                                self._qdrant_stores[Modality.TEXT])]
        else:
            stores_to_search = list(self._qdrant_stores.values())

        filter_cond = {}
        if session_id:
            filter_cond["session_id"] = session_id

        results = []
        for store in stores_to_search:
            qdrant_results = store.search(
                query_vector=query_vec,
                limit=limit,
                filter_conditions=filter_cond if filter_cond else None
            )
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
        """时间近因性得分（指数衰减模型）"""
        try:
            memory_time = datetime.fromisoformat(timestamp)
            current_time = datetime.now()
            age_hours = (current_time - memory_time).total_seconds() / 3600
            decay_factor = 0.1
            recency_score = math.exp(-decay_factor * age_hours / 24)
            return max(0.1, recency_score)
        except Exception:
            return 0.5

    def forget(self, strategy: str = "importance", threshold: float = 0.3,
               older_than_hours: Optional[int] = None,
               capacity_ratio: float = 0.9) -> int:
        removed = 0
        if strategy == "importance":
            to_delete = self._doc_store.query(
                memory_type="perceptual", min_importance=0.0, limit=10000
            )
            ids = [d["id"] for d in to_delete if d["importance"] < threshold]
            if ids:
                self._delete_from_qdrant(ids)
                removed = self._doc_store.delete_by_ids(ids)
        elif strategy == "time" and older_than_hours:
            to_delete = self._doc_store.query_older_than(
                hours=older_than_hours, memory_type="perceptual"
            )
            ids = [d["id"] for d in to_delete]
            if ids:
                self._delete_from_qdrant(ids)
                removed = self._doc_store.delete_by_ids(ids)
        elif strategy == "capacity":
            total = self._doc_store.count(memory_type="perceptual")
            limit_count = int(total * capacity_ratio)
            excess = total - limit_count
            if excess > 0:
                lowest = self._doc_store.get_lowest_importance(
                    limit=excess, memory_type="perceptual"
                )
                ids = [d["id"] for d in lowest]
                if ids:
                    self._delete_from_qdrant(ids)
                    removed = self._doc_store.delete_by_ids(ids)
        return removed

    def _delete_from_qdrant(self, ids: list[str]):
        for store in self._qdrant_stores.values():
            store.delete(ids)

    def count(self) -> int:
        return self._doc_store.count(memory_type="perceptual")

    def clear(self) -> None:
        self._doc_store.clear(memory_type="perceptual")
        for store in self._qdrant_stores.values():
            store.clear()
