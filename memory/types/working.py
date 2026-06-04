﻿# -*- coding: utf-8 -*-
"""
工作记忆（Working Memory）
纯内存存储 + TTL 自动清理 + TF-IDF 混合检索
评分公式: (相似度 x 时间衰减) x (0.8 + 重要性 x 0.4)
"""
import math
import threading
import time
from typing import Optional
from ..base import BaseMemory, MemoryItem, MemorySearchResult, MemoryConfig
from ..embedding import EmbeddingService


class WorkingMemory(BaseMemory):
    """工作记忆：纯内存 + TTL"""

    def __init__(self, config: MemoryConfig):
        super().__init__(config)
        self._items: dict[str, MemoryItem] = {}
        self._lock = threading.Lock()
        self._embedding = EmbeddingService(provider=config.embedding_provider,
                                           vector_size=config.qdrant_vector_size)
        self._cleanup_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._start_cleanup()

    def _start_cleanup(self):
        """启动后台清理线程"""
        def _cleanup_loop():
            while not self._stop_event.is_set():
                self._stop_event.wait(60)
                if not self._stop_event.is_set():
                    self._evict_expired()
        self._cleanup_thread = threading.Thread(target=_cleanup_loop, daemon=True)
        self._cleanup_thread.start()

    def _evict_expired(self):
        """驱逐过期项"""
        with self._lock:
            now = time.time()
            expired = []
            for item_id, item in list(self._items.items()):
                try:
                    item_time = time.mktime(
                        time.strptime(item.timestamp, "%Y-%m-%d %H:%M:%S")
                    )
                    if now - item_time > self.config.working_ttl_seconds:
                        expired.append(item_id)
                except Exception:
                    expired.append(item_id)
            for item_id in expired:
                del self._items[item_id]

    def add(self, item: MemoryItem) -> str:
        with self._lock:
            if len(self._items) >= self.config.working_capacity:
                self._evict_lowest()
            self._items[item.id] = item
            return item.id

    def _evict_lowest(self):
        """驱逐重要性最低的项"""
        if not self._items:
            return
        lowest_id = min(self._items.keys(),
                        key=lambda k: self._items[k].importance)
        del self._items[lowest_id]

    def search(self, query: str, limit: int = 5, min_importance: float = 0.1,
               session_id: Optional[str] = None) -> list[MemorySearchResult]:
        with self._lock:
            candidates = list(self._items.values())

        if session_id:
            candidates = [c for c in candidates if c.session_id == session_id]
        candidates = [c for c in candidates if c.importance >= min_importance]
        if not candidates:
            return []

        query_vec = self._embedding._tfidf_embed(query)
        results = []
        for item in candidates:
            item_vec = self._embedding._tfidf_embed(item.content)
            similarity = EmbeddingService.cosine_similarity(query_vec, item_vec)
            time_decay = self._calculate_time_decay(item.timestamp)
            score = (similarity * time_decay) * (0.8 + item.importance * 0.4)
            if score > 0:
                results.append(MemorySearchResult(item=item, score=score))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    def _calculate_time_decay(self, timestamp: str) -> float:
        """时间衰减因子"""
        try:
            item_time = time.mktime(time.strptime(timestamp, "%Y-%m-%d %H:%M:%S"))
            age_seconds = time.time() - item_time
            age_hours = age_seconds / 3600
            decay = math.exp(-0.1 * age_hours / 24)
            return max(0.1, decay)
        except Exception:
            return 0.5

    def forget(self, strategy: str = "importance", threshold: float = 0.3,
               older_than_hours: Optional[int] = None,
               capacity_ratio: float = 0.9) -> int:
        removed = 0
        with self._lock:
            if strategy == "importance":
                to_remove = [k for k, v in self._items.items()
                             if v.importance < threshold]
                for k in to_remove:
                    del self._items[k]
                    removed += 1
            elif strategy == "time" and older_than_hours:
                now = time.time()
                cutoff = now - older_than_hours * 3600
                to_remove = []
                for k, v in self._items.items():
                    try:
                        t = time.mktime(time.strptime(v.timestamp, "%Y-%m-%d %H:%M:%S"))
                        if t < cutoff:
                            to_remove.append(k)
                    except Exception:
                        to_remove.append(k)
                for k in to_remove:
                    del self._items[k]
                    removed += 1
            elif strategy == "capacity":
                capacity_limit = int(self.config.working_capacity * capacity_ratio)
                while len(self._items) > capacity_limit:
                    lowest_id = min(self._items.keys(),
                                    key=lambda k: self._items[k].importance)
                    del self._items[lowest_id]
                    removed += 1
        return removed

    def count(self) -> int:
        with self._lock:
            return len(self._items)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def get_all(self) -> list[MemoryItem]:
        """获取所有工作记忆（用于 consolidate）"""
        with self._lock:
            return list(self._items.values())

    def close(self):
        """关闭清理线程"""
        self._stop_event.set()
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5)
