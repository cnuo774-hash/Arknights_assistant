# -*- coding: utf-8 -*-
"""
MemoryTool - 记忆系统统一接口
自顶向下设计：「统一入口，分发处理」
对外暴露 add / search / forget / consolidate 四大操作
"""
import os
from typing import Optional, Union
from dotenv import load_dotenv

from .base import (
    MemoryItem, MemorySearchResult, MemoryConfig,
    MemoryType, Modality, BaseMemory
)
from .manager import MemoryManager

load_dotenv()


class MemoryTool:
    """
    记忆系统统一接口
    使用方式:
        tool = MemoryTool()
        tool.add("用户询问了明日方舟的干员推荐", session_id="user_001")
        results = tool.search("干员推荐", memory_type="working")
        tool.forget("importance", threshold=0.2)
        tool.consolidate()
    """

    def __init__(self, config: Optional[MemoryConfig] = None):
        if config is None:
            config = self._build_default_config()
        self.config = config
        self._manager = MemoryManager(config)

    @staticmethod
    def _build_default_config() -> MemoryConfig:
        """从环境变量构建默认配置"""
        return MemoryConfig(
            working_enabled=True,
            working_capacity=100,
            working_ttl_seconds=3600,
            episodic_enabled=True,
            episodic_db_path="./memory_db/episodic.db",
            semantic_enabled=bool(os.getenv("NEO4J_URI", "")),
            neo4j_uri=os.getenv("NEO4J_URI", ""),
            neo4j_username=os.getenv("NEO4J_USERNAME", ""),
            neo4j_password=os.getenv("NEO4J_PASSWORD", ""),
            perceptual_enabled=False,
            perceptual_db_path="./memory_db/perceptual.db",
            qdrant_url=os.getenv("QDRANT_URL", ""),
            qdrant_api_key=os.getenv("QDRANT_API_KEY", ""),
            qdrant_vector_size=1024,
            embedding_provider="dashscope",
        )

    # ---------- 操作1: add ----------

    def add(self, content: str, session_id: str = "",
            memory_type: Union[str, MemoryType] = "working",
            importance: float = 0.5, modality: Union[str, Modality] = "text",
            metadata: Optional[dict] = None) -> str:
        """
        添加记忆
        Args:
            content: 记忆内容
            session_id: 会话ID（可选，自动管理）
            memory_type: 记忆类型 (working/episodic/semantic/perceptual)
            importance: 重要性 (0.0-1.0，默认 0.5)
            modality: 模态 (text/image/audio/video，默认 text)
            metadata: 额外元数据
        Returns:
            记忆ID
        """
        return self._manager.add(
            content=content, session_id=session_id,
            memory_type=memory_type, importance=importance,
            modality=modality, metadata=metadata
        )

    # ---------- 操作2: search ----------

    def search(self, query: str,
               memory_type: Optional[Union[str, MemoryType]] = None,
               memory_types: Optional[list[Union[str, MemoryType]]] = None,
               limit: int = 5, min_importance: float = 0.1,
               session_id: Optional[str] = None) -> list[MemorySearchResult]:
        """
        检索记忆
        Args:
            query: 查询文本
            memory_type: 单种记忆类型
            memory_types: 多种记忆类型
            limit: 返回数量上限
            min_importance: 最低重要性阈值 (默认 0.1)
            session_id: 按会话过滤
        Returns:
            MemorySearchResult 列表（按分数降序）
        """
        return self._manager.search(
            query=query, memory_type=memory_type,
            memory_types=memory_types, limit=limit,
            min_importance=min_importance, session_id=session_id
        )

    # ---------- 操作3: forget ----------

    def forget(self, strategy: str = "importance", threshold: float = 0.3,
               older_than_hours: Optional[int] = None,
               capacity_ratio: float = 0.9,
               memory_type: Optional[Union[str, MemoryType]] = None) -> dict[str, int]:
        """
        遗忘记忆
        Args:
            strategy: 策略 (importance/time/capacity)
            threshold: 重要性阈值（importance 策略）
            older_than_hours: 时间阈值小时（time 策略）
            capacity_ratio: 容量比例（capacity 策略）
            memory_type: 限定记忆类型（None 表示所有）
        Returns:
            {memory_type: deleted_count}
        """
        return self._manager.forget(
            strategy=strategy, threshold=threshold,
            older_than_hours=older_than_hours,
            capacity_ratio=capacity_ratio,
            memory_type=memory_type
        )

    # ---------- 操作4: consolidate ----------

    def consolidate(self, importance_threshold: float = 0.7,
                    source_type: Union[str, MemoryType] = "working",
                    target_type: Union[str, MemoryType] = "episodic") -> int:
        """
        记忆固化：将高重要性短期记忆转为长期记忆
        Args:
            importance_threshold: 重要性阈值 (默认 0.7)
            source_type: 来源记忆类型 (默认 working)
            target_type: 目标记忆类型 (默认 episodic)
        Returns:
            固化的记忆数量
        """
        if isinstance(source_type, str):
            source_type = MemoryType(source_type)
        if isinstance(target_type, str):
            target_type = MemoryType(target_type)
        return self._manager.consolidate(
            importance_threshold=importance_threshold,
            source_type=source_type, target_type=target_type
        )

    # ---------- 统计 ----------

    def get_stats(self) -> dict:
        """获取各类型记忆数量"""
        return self._manager.get_stats()

    def close(self):
        """关闭资源"""
        self._manager.close()
