"""
Embedding服务模块
使用text-embedding-bge-m3模型进行文本向量化
"""
import logging
from typing import List, Union
import requests
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingService:
    """文本向量化服务"""

    def __init__(self, api_config: dict):
        """
        初始化Embedding服务

        Args:
            api_config: API配置，包含 api_base, api_key, embedding_model
        """
        self.api_base = api_config.get("api_base", "http://host.docker.internal:3120")
        self.api_key = api_config.get("api_key", "")
        self.model = api_config.get("embedding_model", "text-embedding-bge-m3")
        self.timeout = api_config.get("timeout", 30)

        # 缓存
        self._cache = {}
        self._cache_size = 1000

    def encode(self, texts: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """
        编码文本为向量

        Args:
            texts: 单个文本或文本列表

        Returns:
            向量或向量列表
        """
        # 处理单个文本
        if isinstance(texts, str):
            return self._encode_single(texts)

        # 处理文本列表
        return self._encode_batch(texts)

    def encode_query(self, text: str) -> np.ndarray:
        """
        编码查询文本(可能有不同的处理)

        Args:
            text: 查询文本

        Returns:
            查询向量
        """
        return self._encode_single(text)

    def batch_encode(self, texts_list: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """
        批量编码文本

        Args:
            texts_list: 文本列表
            batch_size: 批次大小

        Returns:
            向量列表
        """
        all_embeddings = []

        for i in range(0, len(texts_list), batch_size):
            batch = texts_list[i:i + batch_size]
            embeddings = self._encode_batch(batch)
            all_embeddings.extend(embeddings)

        return all_embeddings

    def _encode_single(self, text: str) -> np.ndarray:
        """编码单个文本"""
        # 检查缓存
        cache_key = self._get_cache_key(text)
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            # 调用API
            embedding = self._call_embedding_api([text])[0]

            # 缓存结果
            self._add_to_cache(cache_key, embedding)

            return embedding

        except Exception as e:
            logger.error(f"编码文本失败: {e}")
            # 返回零向量作为降级
            return np.zeros(1024)  # bge-m3的向量维度

    def _encode_batch(self, texts: List[str]) -> List[np.ndarray]:
        """编码文本批次"""
        try:
            return self._call_embedding_api(texts)

        except Exception as e:
            logger.error(f"批量编码失败: {e}")
            # 返回零向量列表
            return [np.zeros(1024) for _ in texts]

    def _call_embedding_api(self, texts: List[str]) -> List[np.ndarray]:
        """
        调用Embedding API

        Args:
            texts: 文本列表

        Returns:
            向量列表
        """
        try:
            # 构建请求
            url = f"{self.api_base}/v1/embeddings"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            payload = {
                "model": self.model,
                "input": texts
            }

            # 发送请求
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=self.timeout
            )

            if response.status_code != 200:
                logger.error(f"Embedding API返回错误: {response.status_code} - {response.text}")
                raise Exception(f"API错误: {response.status_code}")

            # 解析响应
            data = response.json()
            embeddings = []

            for item in data.get("data", []):
                embedding = item.get("embedding", [])
                embeddings.append(np.array(embedding, dtype=np.float32))

            return embeddings

        except requests.Timeout:
            logger.error(f"Embedding API请求超时")
            raise
        except Exception as e:
            logger.error(f"调用Embedding API失败: {e}")
            raise

    def get_embedding_dim(self) -> int:
        """获取向量维度"""
        # bge-m3的向量维度是1024
        if "bge-m3" in self.model.lower():
            return 1024
        # 其他模型可以根据需要添加
        return 1536  # 默认维度(如OpenAI)

    def _get_cache_key(self, text: str) -> str:
        """生成缓存键"""
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()

    def _add_to_cache(self, key: str, embedding: np.ndarray):
        """添加到缓存"""
        # 如果缓存满了，删除最老的一半
        if len(self._cache) >= self._cache_size:
            keys_to_remove = list(self._cache.keys())[:self._cache_size // 2]
            for k in keys_to_remove:
                del self._cache[k]

        self._cache[key] = embedding

    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
        logger.info("Embedding缓存已清空")

    def get_cache_stats(self) -> dict:
        """获取缓存统计"""
        return {
            "cache_size": len(self._cache),
            "max_cache_size": self._cache_size
        }
