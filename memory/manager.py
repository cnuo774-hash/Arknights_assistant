# -*- coding: utf-8 -*-
"""
记忆管理器（MemoryManager）
统一协调调度四种记忆类型，提供 add/search/forget/consolidate 操作
"""
from typing import Optional, Union
from .base import (
    MemoryItem, MemorySearchResult, MemoryConfig,
    MemoryType, Modality, BaseMemory
)
from .types import WorkingMemory, EpisodicMemory, SemanticMemory, PerceptualMemory
from .embedding import EmbeddingService


class MemoryManager:
    """记忆管理器：统一入口，分发处理"""

    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        self._memories: dict[MemoryType, BaseMemory] = {}
        self._embedding = EmbeddingService(
            provider=self.config.embedding_provider,
            vector_size=self.config.qdrant_vector_size
        )
        self._init_errors: dict[str, str] = {}
        self._init_memories()

    def _init_memories(self):
        """根据配置初始化记忆模块（容错处理）"""
        if self.config.working_enabled:
            try:
                self._memories[MemoryType.WORKING] = WorkingMemory(self.config)
            except Exception as e:
                self._init_errors["working"] = str(e)
        if self.config.episodic_enabled:
            try:
                self._memories[MemoryType.EPISODIC] = EpisodicMemory(self.config)
            except Exception as e:
                self._init_errors["episodic"] = str(e)
                print(f"[MemoryManager] 情景记忆初始化失败: {e}")
        if self.config.semantic_enabled:
            try:
                self._memories[MemoryType.SEMANTIC] = SemanticMemory(self.config)
            except Exception as e:
                self._init_errors["semantic"] = str(e)
                print(f"[MemoryManager] 语义记忆初始化失败: {e}")
        if self.config.perceptual_enabled:
            try:
                self._memories[MemoryType.PERCEPTUAL] = PerceptualMemory(self.config)
            except Exception as e:
                self._init_errors["perceptual"] = str(e)
                print(f"[MemoryManager] 感知记忆初始化失败: {e}")

    @property
    def enabled_types(self) -> list[MemoryType]:
        return list(self._memories.keys())

    # ---------- add ----------

    def add(self, content: str, session_id: str = "",
            memory_type: Union[str, MemoryType] = MemoryType.WORKING,
            importance: float = 0.5, modality: Union[str, Modality] = Modality.TEXT,
            metadata: Optional[dict] = None) -> str:
        if isinstance(memory_type, str):
            memory_type = MemoryType(memory_type)
        if isinstance(modality, str):
            modality = Modality(modality)

        item = MemoryItem(
            content=content,
            memory_type=memory_type,
            session_id=session_id,
            importance=max(0.0, min(1.0, importance)),
            modality=modality,
            metadata=metadata or {},
        )

        memory = self._memories.get(memory_type)
        if not memory:
            err = self._init_errors.get(memory_type.value, "未启用")
            raise ValueError(f"记忆类型不可用: {memory_type.value} ({err})")
        return memory.add(item)

    # ---------- search ----------

    def search(self, query: str, memory_type: Optional[Union[str, MemoryType]] = None,
               memory_types: Optional[list[Union[str, MemoryType]]] = None,
               limit: int = 5, min_importance: float = 0.1,
               session_id: Optional[str] = None) -> list[MemorySearchResult]:
        types_to_search = self._resolve_types(memory_type, memory_types)
        all_results = []
        for mt in types_to_search:
            memory = self._memories.get(mt)
            if memory:
                try:
                    results = memory.search(
                        query=query, limit=limit,
                        min_importance=min_importance,
                        session_id=session_id
                    )
                    all_results.extend(results)
                except Exception as e:
                    print(f"[MemoryManager] 检索 {mt.value} 失败: {e}")

        all_results.sort(key=lambda r: r.score, reverse=True)
        return all_results[:limit]

    def _resolve_types(self, memory_type, memory_types) -> list[MemoryType]:
        types = []
        if memory_types:
            for mt in memory_types:
                if isinstance(mt, str):
                    types.append(MemoryType(mt))
                else:
                    types.append(mt)
        elif memory_type:
            if isinstance(memory_type, str):
                types.append(MemoryType(memory_type))
            else:
                types.append(memory_type)
        else:
            types = list(self._memories.keys())
        return types

    # ---------- forget ----------

    def forget(self, strategy: str = "importance", threshold: float = 0.3,
               older_than_hours: Optional[int] = None,
               capacity_ratio: float = 0.9,
               memory_type: Optional[Union[str, MemoryType]] = None) -> dict[str, int]:
        result = {}
        if memory_type:
            mt = MemoryType(memory_type) if isinstance(memory_type, str) else memory_type
            memory = self._memories.get(mt)
            if memory:
                result[mt.value] = memory.forget(
                    strategy=strategy, threshold=threshold,
                    older_than_hours=older_than_hours,
                    capacity_ratio=capacity_ratio
                )
        else:
            for mt, memory in self._memories.items():
                try:
                    result[mt.value] = memory.forget(
                        strategy=strategy, threshold=threshold,
                        older_than_hours=older_than_hours,
                        capacity_ratio=capacity_ratio
                    )
                except Exception as e:
                    print(f"[MemoryManager] 遗忘 {mt.value} 失败: {e}")
                    result[mt.value] = 0
        return result

    # ---------- consolidate ----------

    def consolidate(self, importance_threshold: float = 0.7,
                    source_type: MemoryType = MemoryType.WORKING,
                    target_type: MemoryType = MemoryType.EPISODIC) -> int:
        source = self._memories.get(source_type)
        target = self._memories.get(target_type)
        if not source or not target:
            return 0

        consolidated = 0
        if source_type == MemoryType.WORKING:
            items = source.get_all()
            for item in items:
                if item.importance >= importance_threshold:
                    item.memory_type = target_type
                    target.add(item)
                    consolidated += 1
        return consolidated

    # ---------- 统计 ----------

    def get_stats(self) -> dict:
        stats = {}
        for mt, memory in self._memories.items():
            stats[mt.value] = memory.count()
        return stats

    def close(self):
        for memory in self._memories.values():
            if hasattr(memory, 'close'):
                memory.close()
