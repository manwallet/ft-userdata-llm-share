"""
向量存储模块
使用ChromaDB进行向量存储和检索
"""
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    """向量数据库封装(ChromaDB)"""

    def __init__(self, storage_path: str, embedding_service):
        """
        初始化向量存储

        Args:
            storage_path: 存储路径
            embedding_service: Embedding服务实例
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.embedding_service = embedding_service

        # 初始化ChromaDB
        self._init_chromadb()

    def _init_chromadb(self):
        """初始化ChromaDB客户端"""
        try:
            import chromadb
            from chromadb.config import Settings

            # 创建持久化客户端
            self.client = chromadb.PersistentClient(
                path=str(self.storage_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            logger.info(f"ChromaDB已初始化，存储路径: {self.storage_path}")

        except ImportError:
            logger.error("ChromaDB未安装，请运行: pip install chromadb")
            raise
        except Exception as e:
            logger.error(f"初始化ChromaDB失败: {e}")
            raise

    def _convert_filters_to_chroma_format(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        转换filters为ChromaDB where格式

        ChromaDB要求where条件使用运算符，如:
        - $eq: 等于
        - $ne: 不等于
        - $gt: 大于
        - $gte: 大于等于
        - $lt: 小于
        - $lte: 小于等于
        - $and: AND逻辑
        - $or: OR逻辑

        Args:
            filters: 原始过滤条件，如 {'pair': 'BTC/USDT:USDT', 'action': 'entry'}

        Returns:
            ChromaDB格式的where条件
        """
        if not filters:
            return None

        # 如果已经是ChromaDB格式（包含运算符），直接返回
        if any(k.startswith('$') for k in filters.keys()):
            return filters

        # 转换简单的key-value格式
        chroma_conditions = []
        for key, value in filters.items():
            if isinstance(value, dict):
                # 如果value已经是运算符格式，直接使用
                if any(k.startswith('$') for k in value.keys()):
                    chroma_conditions.append({key: value})
                else:
                    # 否则转换为$eq
                    chroma_conditions.append({key: {"$eq": value}})
            else:
                # 简单值转换为$eq运算符
                chroma_conditions.append({key: {"$eq": value}})

        # 如果只有一个条件，直接返回
        if len(chroma_conditions) == 1:
            return chroma_conditions[0]

        # 多个条件使用$and连接
        return {"$and": chroma_conditions}

    def get_or_create_collection(self, collection_name: str) -> Any:
        """
        获取或创建collection

        Args:
            collection_name: collection名称

        Returns:
            Collection对象
        """
        try:
            collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
            )
            return collection

        except Exception as e:
            logger.error(f"获取collection失败: {e}")
            raise

    def add_documents(
        self,
        collection_name: str,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> bool:
        """
        添加文档到向量库

        Args:
            collection_name: collection名称
            documents: 文档文本列表
            metadatas: 元数据列表
            ids: 文档ID列表(可选)

        Returns:
            是否成功
        """
        try:
            collection = self.get_or_create_collection(collection_name)

            # 生成embeddings
            embeddings = self.embedding_service.batch_encode(documents)

            # 生成IDs(如果未提供)
            if ids is None:
                import uuid
                ids = [str(uuid.uuid4()) for _ in documents]

            # 转换embeddings为列表格式
            embeddings_list = [emb.tolist() if isinstance(emb, np.ndarray) else emb for emb in embeddings]

            # 添加到collection
            collection.add(
                documents=documents,
                embeddings=embeddings_list,
                metadatas=metadatas,
                ids=ids
            )

            logger.debug(f"已添加 {len(documents)} 条文档到 {collection_name}")
            return True

        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            return False

    def search(
        self,
        collection_name: str,
        query_text: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        搜索相似文档

        Args:
            collection_name: collection名称
            query_text: 查询文本
            top_k: 返回数量
            filters: 过滤条件

        Returns:
            搜索结果列表
        """
        try:
            collection = self.get_or_create_collection(collection_name)

            # 生成查询向量
            query_embedding = self.embedding_service.encode_query(query_text)

            # 转换为列表
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()

            # 转换filters为ChromaDB格式
            # ChromaDB要求where条件必须使用运算符（如$eq, $gte等）
            chroma_where = None
            if filters:
                chroma_where = self._convert_filters_to_chroma_format(filters)

            # 执行搜索
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=chroma_where
            )

            # 格式化结果
            formatted_results = []
            if results and results['ids']:
                for i in range(len(results['ids'][0])):
                    formatted_results.append({
                        "id": results['ids'][0][i],
                        "document": results['documents'][0][i] if results['documents'] else "",
                        "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                        "similarity": 1 - results['distances'][0][i] if results['distances'] else 0,  # 转换距离为相似度
                        "distance": results['distances'][0][i] if results['distances'] else 0
                    })

            return formatted_results

        except Exception as e:
            logger.error(f"搜索文档失败: {e}")
            return []

    def delete_documents(
        self,
        collection_name: str,
        ids: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        删除文档

        Args:
            collection_name: collection名称
            ids: 文档ID列表
            filters: 过滤条件

        Returns:
            是否成功
        """
        try:
            collection = self.get_or_create_collection(collection_name)

            if ids:
                collection.delete(ids=ids)
            elif filters:
                collection.delete(where=filters)
            else:
                logger.warning("未提供删除条件")
                return False

            logger.info(f"已从 {collection_name} 删除文档")
            return True

        except Exception as e:
            logger.error(f"删除文档失败: {e}")
            return False

    def delete_old_records(
        self,
        collection_name: str,
        days: int
    ) -> int:
        """
        删除旧记录

        Args:
            collection_name: collection名称
            days: 保留天数

        Returns:
            删除数量
        """
        try:
            from datetime import datetime, timedelta

            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

            collection = self.get_or_create_collection(collection_name)

            # 查询旧记录
            results = collection.get(
                where={"timestamp": {"$lt": cutoff_date}}
            )

            if results and results['ids']:
                # 删除
                collection.delete(ids=results['ids'])
                count = len(results['ids'])
                logger.info(f"已删除 {count} 条超过 {days} 天的记录")
                return count

            return 0

        except Exception as e:
            logger.error(f"删除旧记录失败: {e}")
            return 0

    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """
        获取collection统计信息

        Args:
            collection_name: collection名称

        Returns:
            统计信息
        """
        try:
            collection = self.get_or_create_collection(collection_name)

            count = collection.count()

            return {
                "name": collection_name,
                "count": count,
                "storage_path": str(self.storage_path)
            }

        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {"name": collection_name, "count": 0, "error": str(e)}

    def list_collections(self) -> List[str]:
        """列出所有collections"""
        try:
            collections = self.client.list_collections()
            return [col.name for col in collections]

        except Exception as e:
            logger.error(f"列出collections失败: {e}")
            return []

    def reset_collection(self, collection_name: str) -> bool:
        """
        重置(清空)collection

        Args:
            collection_name: collection名称

        Returns:
            是否成功
        """
        try:
            self.client.delete_collection(name=collection_name)
            logger.info(f"Collection {collection_name} 已重置")
            return True

        except Exception as e:
            logger.error(f"重置collection失败: {e}")
            return False

    def backup(self, backup_path: str) -> bool:
        """
        备份向量数据库

        Args:
            backup_path: 备份路径

        Returns:
            是否成功
        """
        try:
            import shutil
            backup_path = Path(backup_path)
            backup_path.mkdir(parents=True, exist_ok=True)

            # 复制数据库文件
            shutil.copytree(
                self.storage_path,
                backup_path / "vector_store_backup",
                dirs_exist_ok=True
            )

            logger.info(f"向量数据库已备份到 {backup_path}")
            return True

        except Exception as e:
            logger.error(f"备份失败: {e}")
            return False
