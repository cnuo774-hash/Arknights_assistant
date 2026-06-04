# -*- coding: utf-8 -*-
"""
聊天历史存储（桥接层）
保持与 LangChain RunnableWithMessageHistory 的兼容接口，
内部纯依赖 MemoryTool 记忆模块。
"""
from typing import Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory
from memory import MemoryTool, MemoryType
from memory.storage.document_store import DocumentStore

_memory_tool: MemoryTool = None


def _get_tool() -> MemoryTool:
    global _memory_tool
    if _memory_tool is None:
        _memory_tool = MemoryTool()
    return _memory_tool


def get_history(session_id: str) -> "FileChatMessageHistory":
    return FileChatMessageHistory(session_id)


class FileChatMessageHistory(BaseChatMessageHistory):
    """基于 MemoryTool 的聊天历史管理"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._tool = _get_tool()
        self._doc_store = DocumentStore(
            db_path="./memory_db/episodic.db",
            table_name="episodic_memories"
        )

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        for msg in messages:
            content = self._msg_to_text(msg)
            importance = 0.6 if isinstance(msg, AIMessage) else 0.5
            try:
                self._tool.add(
                    content=content,
                    session_id=self.session_id,
                    memory_type=MemoryType.WORKING,
                    importance=importance,
                )
            except Exception:
                pass
            try:
                self._tool.add(
                    content=content,
                    session_id=self.session_id,
                    memory_type=MemoryType.EPISODIC,
                    importance=importance,
                )
            except Exception:
                pass

    @property
    def messages(self) -> list[BaseMessage]:
        try:
            rows = self._doc_store.query(
                session_id=self.session_id,
                memory_type="episodic",
                order_by="timestamp ASC",
                limit=200,
            )
            if rows:
                return self._rows_to_messages(rows)
        except Exception:
            pass
        return []

    def clear(self) -> None:
        try:
            self._tool.forget(
                strategy="time",
                older_than_hours=0,
                memory_type=MemoryType.WORKING,
            )
        except Exception:
            pass

    def _rows_to_messages(self, rows: list[dict]) -> list[BaseMessage]:
        messages = []
        for row in rows:
            content = row["content"]
            imp = row.get("importance", 0.5)
            if content.startswith("Human: "):
                messages.append(HumanMessage(content=content[7:]))
            elif content.startswith("AI: "):
                messages.append(AIMessage(content=content[4:]))
            else:
                if imp >= 0.55:
                    messages.append(AIMessage(content=content))
                else:
                    messages.append(HumanMessage(content=content))
        return messages

    @staticmethod
    def _msg_to_text(msg: BaseMessage) -> str:
        if isinstance(msg, HumanMessage):
            return f"Human: {msg.content}"
        elif isinstance(msg, AIMessage):
            return f"AI: {msg.content}"
        return str(msg.content)
