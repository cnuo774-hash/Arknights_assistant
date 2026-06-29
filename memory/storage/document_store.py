# -*- coding: utf-8 -*-
"""
SQLite 文档存储后端（结构化持久化）
"""
import sqlite3
import os
import json
from typing import Optional


class DocumentStore:
    """SQLite 文档存储封装"""

    def __init__(self, db_path: str, table_name: str = "memories"):
        self.db_path = db_path
        self.table_name = table_name
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_table()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_table(self):
        with self._get_conn() as conn:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    memory_type TEXT NOT NULL DEFAULT 'working',
                    session_id TEXT NOT NULL DEFAULT '',
                    importance REAL NOT NULL DEFAULT 0.5,
                    modality TEXT NOT NULL DEFAULT 'text',
                    timestamp TEXT NOT NULL,
                    metadata TEXT DEFAULT '{{}}'
                )
            """)
            conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_session
                ON {self.table_name}(session_id)
            """)
            conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_importance
                ON {self.table_name}(importance)
            """)
            conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_timestamp
                ON {self.table_name}(timestamp)
            """)
            conn.commit()

    def insert(self, item: dict) -> None:
        """插入文档"""
        with self._get_conn() as conn:
            conn.execute(f"""
                INSERT OR REPLACE INTO {self.table_name}
                (id, content, memory_type, session_id, importance,
                 modality, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                item["id"],
                item["content"],
                item.get("memory_type", "working"),
                item.get("session_id", ""),
                item.get("importance", 0.5),
                item.get("modality", "text"),
                item.get("timestamp", ""),
                json.dumps(item.get("metadata", {}), ensure_ascii=False),
            ))
            conn.commit()

    def get_by_id(self, item_id: str) -> Optional[dict]:
        """按 ID 查询"""
        with self._get_conn() as conn:
            row = conn.execute(
                f"SELECT * FROM {self.table_name} WHERE id = ?", (item_id,)
            ).fetchone()
            return dict(row) if row else None

    def query(self, session_id: Optional[str] = None,
              memory_type: Optional[str] = None,
              min_importance: float = 0.0,
              order_by: str = "timestamp DESC",
              limit: int = 100) -> list[dict]:
        """灵活查询"""
        conditions = ["1=1"]
        params = []
        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        if memory_type:
            conditions.append("memory_type = ?")
            params.append(memory_type)
        if min_importance > 0:
            conditions.append("importance >= ?")
            params.append(min_importance)

        query_sql = (
            f"SELECT * FROM {self.table_name} WHERE {' AND '.join(conditions)} "
            f"ORDER BY {order_by} LIMIT ?"
        )
        params.append(limit)

        with self._get_conn() as conn:
            rows = conn.execute(query_sql, params).fetchall()
            return [dict(row) for row in rows]

    def query_older_than(self, hours: int, min_importance: float = 0.0,
                         memory_type: Optional[str] = None) -> list[dict]:
        """查询超过指定时间的记录"""
        conditions = [
            f"timestamp < datetime('now', '-{hours} hours')",
        ]
        params = []
        if memory_type:
            conditions.append("memory_type = ?")
            params.append(memory_type)
        if min_importance > 0:
            conditions.append("importance >= ?")
            params.append(min_importance)

        query_sql = f"SELECT * FROM {self.table_name} WHERE {' AND '.join(conditions)}"
        with self._get_conn() as conn:
            rows = conn.execute(query_sql, params).fetchall()
            return [dict(row) for row in rows]

    def delete_by_ids(self, ids: list[str]) -> int:
        """批量删除"""
        if not ids:
            return 0
        with self._get_conn() as conn:
            placeholders = ','.join(['?'] * len(ids))
            cursor = conn.execute(
                f"DELETE FROM {self.table_name} WHERE id IN ({placeholders})", ids
            )
            conn.commit()
            return cursor.rowcount

    def delete_by_importance(self, threshold: float,
                             memory_type: Optional[str] = None) -> int:
        """按重要性删除"""
        conditions = ["importance < ?"]
        params = [threshold]
        if memory_type:
            conditions.append("memory_type = ?")
            params.append(memory_type)
        with self._get_conn() as conn:
            cursor = conn.execute(
                f"DELETE FROM {self.table_name} WHERE {' AND '.join(conditions)}",
                params
            )
            conn.commit()
            return cursor.rowcount

    def delete_older_than(self, hours: int,
                          memory_type: Optional[str] = None) -> int:
        """按时间删除"""
        conditions = [f"timestamp < datetime('now', '-{hours} hours')"]
        params = []
        if memory_type:
            conditions.append("memory_type = ?")
            params.append(memory_type)
        with self._get_conn() as conn:
            cursor = conn.execute(
                f"DELETE FROM {self.table_name} WHERE {' AND '.join(conditions)}",
                params
            )
            conn.commit()
            return cursor.rowcount

    def get_lowest_importance(self, limit: int,
                              memory_type: Optional[str] = None) -> list[dict]:
        """获取重要性最低的记录（用于容量清理）"""
        conditions = ["1=1"]
        params = []
        if memory_type:
            conditions.append("memory_type = ?")
            params.append(memory_type)
        with self._get_conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM {self.table_name} WHERE {' AND '.join(conditions)} "
                f"ORDER BY importance ASC LIMIT ?",
                params + [limit]
            ).fetchall()
            return [dict(row) for row in rows]

    def count(self, session_id: Optional[str] = None,
              memory_type: Optional[str] = None) -> int:
        """计数"""
        conditions = ["1=1"]
        params = []
        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        if memory_type:
            conditions.append("memory_type = ?")
            params.append(memory_type)
        with self._get_conn() as conn:
            row = conn.execute(
                f"SELECT COUNT(*) FROM {self.table_name} WHERE {' AND '.join(conditions)}",
                params
            ).fetchone()
            return row[0] if row else 0

    def clear(self, memory_type: Optional[str] = None) -> None:
        """清空表"""
        with self._get_conn() as conn:
            if memory_type:
                conn.execute(
                    f"DELETE FROM {self.table_name} WHERE memory_type = ?",
                    (memory_type,)
                )
            else:
                conn.execute(f"DELETE FROM {self.table_name}")
            conn.commit()
