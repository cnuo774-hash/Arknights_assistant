# -*- coding: utf-8 -*-
"""
记忆系统基础模块
定义核心数据结构：MemoryItem, MemoryConfig, BaseMemory
"""
import uuid
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any
from enum import Enum


class MemoryType(str, Enum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PERCEPTUAL = "perceptual"


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


@dataclass
class MemoryItem:
    """记忆单元的核心数据结构"""
    content: str
    memory_type: MemoryType = MemoryType.WORKING
    session_id: str = ""
    importance: float = 0.5
    modality: Modality = Modality.TEXT
    metadata: dict = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    embedding: Optional[list[float]] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "session_id": self.session_id,
            "importance": self.importance,
            "modality": self.modality.value,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryItem":
        return cls(
            id=d.get("id", uuid.uuid4().hex),
            content=d["content"],
            memory_type=MemoryType(d.get("memory_type", "working")),
            session_id=d.get("session_id", ""),
            importance=float(d.get("importance", 0.5)),
            modality=Modality(d.get("modality", "text")),
            timestamp=d.get("timestamp", ""),
            metadata=d.get("metadata", {}),
        )


@dataclass
class MemorySearchResult:
    """记忆检索结果"""
    item: MemoryItem
    score: float


@dataclass
class MemoryConfig:
    """记忆系统全局配置"""
    working_enabled: bool = True
    working_capacity: int = 100
    working_ttl_seconds: int = 3600
    episodic_enabled: bool = True
    episodic_db_path: str = "./memory_db/episodic.db"
    semantic_enabled: bool = False
    neo4j_uri: str = ""
    neo4j_username: str = ""
    neo4j_password: str = ""
    perceptual_enabled: bool = False
    perceptual_db_path: str = "./memory_db/perceptual.db"
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection_prefix: str = "memory"
    qdrant_vector_size: int = 1024
    embedding_provider: str = "dashscope"


class BaseMemory(ABC):
    """所有记忆类型的抽象基类"""

    def __init__(self, config: MemoryConfig):
        self.config = config

    @abstractmethod
    def add(self, item: MemoryItem) -> str:
        """添加记忆，返回记忆ID"""
        ...

    @abstractmethod
    def search(self, query: str, limit: int = 5, min_importance: float = 0.1,
               session_id: Optional[str] = None) -> list[MemorySearchResult]:
        """检索记忆"""
        ...

    @abstractmethod
    def forget(self, strategy: str = "importance", threshold: float = 0.3,
               older_than_hours: Optional[int] = None, capacity_ratio: float = 0.9) -> int:
        """遗忘记忆，返回删除数量"""
        ...

    @abstractmethod
    def count(self) -> int:
        """记忆数量"""
        ...

    @abstractmethod
    def clear(self) -> None:
        """清空所有记忆"""
        ...
