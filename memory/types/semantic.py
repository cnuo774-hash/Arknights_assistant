# -*- coding: utf-8 -*-
"""
语义记忆（Semantic Memory）
Neo4j 图数据库 + Qdrant 向量数据库 混合架构
评分公式: (向量相似度 x 0.7 + 图相似度 x 0.3) x (0.8 + 重要性 x 0.4)
"""
import math
from datetime import datetime, timedelta
from typing import Optional
from ..base import BaseMemory, MemoryItem, MemorySearchResult, MemoryConfig, MemoryType
from ..embedding import EmbeddingService
from ..storage import QdrantStore, Neo4jStore


class SemanticMemory(BaseMemory):
    """语义记忆：Qdrant + Neo4j"""

    def __init__(self, config: MemoryConfig):
        super().__init__(config)
        self._qdrant = QdrantStore(
            url=config.qdrant_url,
            api_key=config.qdrant_api_key,
            collection_name=f"{config.qdrant_collection_prefix}_semantic",
            vector_size=config.qdrant_vector_size
        )
        self._neo4j = Neo4jStore(
            uri=config.neo4j_uri,
            username=config.neo4j_username,
            password=config.neo4j_password
        )
        self._embedding = EmbeddingService(
            provider=config.embedding_provider,
            vector_size=config.qdrant_vector_size
        )
        self._graph_score_cache: dict[str, float] = {}

    def add(self, item: MemoryItem) -> str:
        item.memory_type = MemoryType.SEMANTIC
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
        self._neo4j.create_memory_node(
            item_id=item.id,
            content=item.content,
            importance=item.importance,
            session_id=item.session_id
        )
        entities = self._neo4j.extract_entities(item.content)
        for entity, relation in entities:
            self._neo4j.create_entity_relationship(
                source_id=item.id,
                target_entity=entity,
                relation=relation
            )
        return item.id

    def search(self, query: str, limit: int = 5, min_importance: float = 0.1,
               session_id: Optional[str] = None) -> list[MemorySearchResult]:
        query_vec = self._embedding.embed(query)
        filter_cond = {}
        if session_id:
            filter_cond["session_id"] = session_id
        qdrant_results = self._qdrant.search(
            query_vector=query_vec,
            limit=limit * 2,
            filter_conditions=filter_cond if filter_cond else None
        )

        graph_results = self._neo4j.search_by_graph(query, limit=limit * 2)
        graph_scores: dict[str, float] = {}
        max_entity_count = max((r["entity_count"] for r in graph_results), default=1)
        for r in graph_results:
            graph_scores[r["id"]] = min(r["entity_count"] / max_entity_count, 1.0)

        results = []
        for scored in qdrant_results:
            item_id = scored.id
            payload = scored.payload or {}
            importance = payload.get("importance", 0.5)
            if importance < min_importance:
                continue

            vec_sim = scored.score
            graph_sim = graph_scores.get(str(item_id), 0.0)
            score = (vec_sim * 0.7 + graph_sim * 0.3) * (0.8 + importance * 0.4)
            results.append(MemorySearchResult(
                item=MemoryItem(
                    id=str(item_id),
                    content=str(payload.get("content", "")),
                    memory_type=MemoryType.SEMANTIC,
                    session_id=str(payload.get("session_id", "")),
                    importance=importance,
                    timestamp=str(payload.get("timestamp", "")),
                ),
                score=score
            ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    def forget(self, strategy: str = "importance", threshold: float = 0.3,
               older_than_hours: Optional[int] = None,
               capacity_ratio: float = 0.9) -> int:
        removed = 0
        if strategy == "importance":
            all_points = self._qdrant.search(
                query_vector=[0.0] * self.config.qdrant_vector_size,
                limit=1000
            )
            to_delete = []
            for p in all_points:
                imp = (p.payload or {}).get("importance", 0.5)
                if imp < threshold:
                    to_delete.append(str(p.id))
            if to_delete:
                self._qdrant.delete(to_delete)
                for pid in to_delete:
                    self._neo4j.delete_node(pid)
                removed = len(to_delete)
        elif strategy == "time" and older_than_hours:
            all_points = self._qdrant.search(
                query_vector=[0.0] * self.config.qdrant_vector_size,
                limit=1000
            )
            cutoff = datetime.now() - timedelta(hours=older_than_hours)
            to_delete = []
            for p in all_points:
                ts = (p.payload or {}).get("timestamp", "")
                try:
                    if datetime.fromisoformat(ts) < cutoff:
                        to_delete.append(str(p.id))
                except Exception:
                    pass
            if to_delete:
                self._qdrant.delete(to_delete)
                for pid in to_delete:
                    self._neo4j.delete_node(pid)
                removed = len(to_delete)
        return removed

    def count(self) -> int:
        return self._neo4j.count_nodes()

    def clear(self) -> None:
        self._qdrant.clear()
        self._neo4j.clear()

    def close(self):
        self._neo4j.close()
