# -*- coding: utf-8 -*-
"""
Qdrant 向量存储后端（云服务）
"""
import uuid
from typing import Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition,
    MatchValue, Range, ScoredPoint
)


class QdrantStore:
    """Qdrant 向量存储封装"""

    def __init__(self, url: str, api_key: str, collection_name: str,
                 vector_size: int = 1024):
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.client = QdrantClient(url=url, api_key=api_key)
        self._ensure_collection()

    def _ensure_collection(self):
        """确保集合存在"""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )

    def upsert(self, point_id: str, vector: list[float],
               payload: Optional[dict] = None) -> None:
        """插入或更新向量"""
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload or {}
                )
            ]
        )

    def upsert_batch(self, points: list[tuple[str, list[float], dict]]) -> None:
        """批量插入"""
        point_structs = [
            PointStruct(id=pid, vector=vec, payload=payload)
            for pid, vec, payload in points
        ]
        self.client.upsert(
            collection_name=self.collection_name,
            points=point_structs
        )

    def search(self, query_vector: list[float], limit: int = 5,
               filter_conditions: Optional[dict] = None,
               score_threshold: float = 0.0) -> list[ScoredPoint]:
        """向量检索"""
        query_filter = None
        if filter_conditions:
            conditions = []
            for key, value in filter_conditions.items():
                if isinstance(value, (int, float)):
                    conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )
                elif isinstance(value, str):
                    conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )
            if conditions:
                query_filter = Filter(must=conditions)

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            query_filter=query_filter,
            score_threshold=score_threshold,
            with_payload=True,
        )
        return results

    def delete(self, point_ids: list[str]) -> None:
        """删除向量"""
        if point_ids:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=point_ids
            )

    def delete_by_filter(self, filter_conditions: dict) -> None:
        """按条件删除"""
        conditions = []
        for key, value in filter_conditions.items():
            if isinstance(value, str):
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
        if conditions:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(must=conditions)
            )

    def count(self) -> int:
        """集合中的向量数量"""
        info = self.client.get_collection(self.collection_name)
        return info.points_count

    def clear(self) -> None:
        """清空集合"""
        self.client.delete_collection(self.collection_name)
        self._ensure_collection()
