# -*- coding: utf-8 -*-
"""
Neo4j 图存储后端（云服务 - Neo4j Aura）
"""
from typing import Optional
from neo4j import GraphDatabase


class Neo4jStore:
    """Neo4j 图数据库封装"""

    def __init__(self, uri: str, username: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self._init_constraints()

    def _init_constraints(self):
        """初始化约束和索引"""
        with self.driver.session() as session:
            try:
                session.run(
                    "CREATE CONSTRAINT IF NOT EXISTS "
                    "FOR (m:Memory) REQUIRE m.id IS UNIQUE"
                )
            except Exception:
                pass
            try:
                session.run(
                    "CREATE CONSTRAINT IF NOT EXISTS "
                    "FOR (e:Entity) REQUIRE e.name IS UNIQUE"
                )
            except Exception:
                pass

    def create_memory_node(self, item_id: str, content: str,
                           importance: float = 0.5,
                           session_id: str = "") -> None:
        """创建记忆节点"""
        with self.driver.session() as session:
            session.run("""
                MERGE (m:Memory {id: $id})
                SET m.content = $content,
                    m.importance = $importance,
                    m.session_id = $session_id,
                    m.updated_at = datetime()
            """, id=item_id, content=content,
               importance=importance, session_id=session_id)

    def create_entity_relationship(self, source_id: str, target_entity: str,
                                   relation: str = "RELATED_TO") -> None:
        """创建实体关系"""
        with self.driver.session() as session:
            session.run("""
                MATCH (m:Memory {id: $source_id})
                MERGE (e:Entity {name: $target})
                MERGE (m)-[r:RELATED_TO]->(e)
                SET r.type = $relation
            """, source_id=source_id, target=target_entity,
               relation=relation)

    def extract_entities(self, content: str) -> list[tuple[str, str]]:
        """从文本中提取实体和关系（简化版，使用规则匹配）"""
        import re
        entities = []
        book_pattern = r'《([^》]+)》'
        quote_pattern = r'["\u201c]([^"\u201d]+)["\u201d]'
        for match in re.finditer(book_pattern, content):
            entities.append((match.group(1), "MENTIONS"))
        for match in re.finditer(quote_pattern, content):
            entities.append((match.group(1), "QUOTES"))
        return entities

    def search_by_graph(self, query: str, limit: int = 5) -> list[dict]:
        """图检索：查找与查询实体相关的记忆"""
        results = []
        with self.driver.session() as session:
            result = session.run("""
                MATCH (m:Memory)-[r:RELATED_TO]->(e:Entity)
                WHERE e.name CONTAINS $query OR m.content CONTAINS $query
                RETURN m.id AS id, m.content AS content,
                       m.importance AS importance, m.session_id AS session_id,
                       collect(DISTINCT e.name) AS entities,
                       count(DISTINCT e) AS entity_count
                ORDER BY entity_count DESC, m.importance DESC
                LIMIT $limit
            """, query=query, limit=limit)
            for record in result:
                results.append({
                    "id": record["id"],
                    "content": record["content"],
                    "importance": record["importance"],
                    "session_id": record["session_id"],
                    "entities": record["entities"],
                    "entity_count": record["entity_count"],
                })
        return results

    def delete_node(self, item_id: str) -> None:
        """删除记忆节点及其关系"""
        with self.driver.session() as session:
            session.run(
                "MATCH (m:Memory {id: $id}) DETACH DELETE m",
                id=item_id
            )

    def count_nodes(self) -> int:
        """记忆节点数量"""
        with self.driver.session() as session:
            result = session.run("MATCH (m:Memory) RETURN count(m) AS cnt")
            record = result.single()
            return record["cnt"] if record else 0

    def clear(self) -> None:
        """清空所有记忆节点"""
        with self.driver.session() as session:
            session.run("MATCH (m:Memory) DETACH DELETE m")

    def close(self):
        """关闭连接"""
        self.driver.close()
