"""
RAG管理器模块
管理RAG系统的核心逻辑：存储、检索、清理
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class RAGManager:
    """RAG系统管理器"""

    def __init__(self, rag_config: Dict[str, Any], embedding_service, vector_store):
        """
        初始化RAG管理器

        Args:
            rag_config: RAG配置
            embedding_service: Embedding服务
            vector_store: 向量存储
        """
        self.config = rag_config
        self.embedding_service = embedding_service
        self.vector_store = vector_store

        self.enabled = rag_config.get("enable", True)
        self.top_k = rag_config.get("similarity_top_k", 5)
        self.min_similarity = rag_config.get("min_similarity", 0.7)

        # Collection名称
        self.collections = {
            "decisions": "trade_decisions",  # 每次决策
            "trades": "completed_trades",    # 完整交易
            "states": "market_states"        # 市场状态
        }

        logger.info(f"RAG管理器已初始化 (启用: {self.enabled})")

    def store_decision(
        self,
        pair: str,
        action: str,
        market_context: str,
        decision: str,
        reasoning: str,
        confidence: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        存储决策记录

        Args:
            pair: 交易对
            action: 操作类型 (entry_long/entry_short/exit)
            market_context: 市场上下文
            decision: 决策内容
            reasoning: 推理过程
            confidence: 信心度
            metadata: 额外元数据

        Returns:
            是否成功
        """
        if not self.enabled:
            return False

        try:
            # 构建文档
            document = f"""
交易对: {pair}
操作: {action}
市场情况: {market_context}
决策: {decision}
推理: {reasoning}
信心度: {confidence}
"""

            # 构建元数据
            doc_metadata = {
                "pair": pair,
                "action": action,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat(),
                "decision": decision,
                "reasoning": reasoning
            }

            if metadata:
                doc_metadata.update(metadata)

            # 存储到向量库
            success = self.vector_store.add_documents(
                collection_name=self.collections["decisions"],
                documents=[document],
                metadatas=[doc_metadata]
            )

            if success:
                logger.debug(f"已存储决策: {pair} - {action}")

            return success

        except Exception as e:
            logger.error(f"存储决策失败: {e}")
            return False

    def store_trade(
        self,
        pair: str,
        side: str,
        entry_reason: str,
        exit_reason: str,
        profit_pct: float,
        duration_minutes: int,
        entry_price: float,
        exit_price: float,
        leverage: float,
        market_condition: str,
        lessons: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        存储完整交易记录

        Args:
            pair: 交易对
            side: 方向 (long/short)
            entry_reason: 入场理由
            exit_reason: 出场理由
            profit_pct: 盈亏百分比
            duration_minutes: 持仓时长(分钟)
            entry_price: 入场价格
            exit_price: 出场价格
            leverage: 杠杆
            market_condition: 市场状况
            lessons: 经验教训
            metadata: 额外元数据

        Returns:
            是否成功
        """
        if not self.enabled:
            return False

        try:
            # 构建文档
            result = "盈利" if profit_pct > 0 else "亏损"
            document = f"""
交易对: {pair}
方向: {side}
入场理由: {entry_reason}
出场理由: {exit_reason}
结果: {result} {profit_pct:.2f}%
市场状况: {market_condition}
经验教训: {lessons}
持仓时长: {duration_minutes}分钟
杠杆: {leverage}x
"""

            # 构建元数据
            doc_metadata = {
                "pair": pair,
                "side": side,
                "profit_pct": profit_pct,
                "duration_minutes": duration_minutes,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "leverage": leverage,
                "timestamp": datetime.now().isoformat(),
                "entry_reason": entry_reason,
                "exit_reason": exit_reason,
                "market_condition": market_condition,
                "lessons": lessons,
                "result": result
            }

            if metadata:
                doc_metadata.update(metadata)

            # 存储到向量库
            success = self.vector_store.add_documents(
                collection_name=self.collections["trades"],
                documents=[document],
                metadatas=[doc_metadata]
            )

            if success:
                logger.info(f"已存储交易: {pair} {side} {result} {profit_pct:.2f}%")

            return success

        except Exception as e:
            logger.error(f"存储交易失败: {e}")
            return False

    def store_market_state(
        self,
        pair: str,
        state_description: str,
        indicators: Dict[str, float],
        decision_made: str,
        outcome: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        存储市场状态快照

        Args:
            pair: 交易对
            state_description: 状态描述
            indicators: 技术指标
            decision_made: 做出的决策
            outcome: 结果
            metadata: 额外元数据

        Returns:
            是否成功
        """
        if not self.enabled:
            return False

        try:
            # 构建文档
            indicators_str = "\n".join([f"  {k}: {v:.4f}" for k, v in indicators.items()])
            document = f"""
交易对: {pair}
市场状态: {state_description}
技术指标:
{indicators_str}
决策: {decision_made}
结果: {outcome}
"""

            # 构建元数据
            doc_metadata = {
                "pair": pair,
                "timestamp": datetime.now().isoformat(),
                "description": state_description,
                "decision": decision_made,
                "outcome": outcome,
                "indicators": json.dumps(indicators)
            }

            if metadata:
                doc_metadata.update(metadata)

            # 存储
            success = self.vector_store.add_documents(
                collection_name=self.collections["states"],
                documents=[document],
                metadatas=[doc_metadata]
            )

            return success

        except Exception as e:
            logger.error(f"存储市场状态失败: {e}")
            return False

    def search_similar(
        self,
        query: str,
        collection: str = "decisions",
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        min_similarity: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        搜索相似记录

        Args:
            query: 查询文本
            collection: collection类型 (decisions/trades/states)
            top_k: 返回数量
            filters: 过滤条件
            min_similarity: 最小相似度

        Returns:
            相似记录列表
        """
        if not self.enabled:
            return []

        try:
            # 获取collection名称
            collection_name = self.collections.get(collection, collection)

            # 使用默认值
            if top_k is None:
                top_k = self.top_k
            if min_similarity is None:
                min_similarity = self.min_similarity

            # 搜索
            results = self.vector_store.search(
                collection_name=collection_name,
                query_text=query,
                top_k=top_k,
                filters=filters
            )

            # 过滤低相似度结果
            filtered_results = [
                r for r in results
                if r.get("similarity", 0) >= min_similarity
            ]

            logger.debug(f"检索到 {len(filtered_results)}/{len(results)} 条相关记录")

            return filtered_results

        except Exception as e:
            logger.error(f"检索失败: {e}")
            return []

    def get_relevant_context(
        self,
        pair: str,
        current_state: str,
        action_type: str = "entry"
    ) -> str:
        """
        获取相关上下文

        Args:
            pair: 交易对
            current_state: 当前状态描述
            action_type: 操作类型

        Returns:
            格式化的上下文字符串
        """
        if not self.enabled:
            return "RAG系统未启用"

        try:
            # 构建查询
            query = f"{pair} {current_state} {action_type}"

            # 检索相关决策
            decisions = self.search_similar(
                query=query,
                collection="decisions",
                top_k=3,
                filters={"pair": pair}
            )

            # 检索相关交易
            trades = self.search_similar(
                query=query,
                collection="trades",
                top_k=2,
                filters={"pair": pair}
            )

            # 构建上下文
            context_parts = []

            if decisions:
                context_parts.append("相关历史决策:")
                for i, dec in enumerate(decisions, 1):
                    metadata = dec.get("metadata", {})
                    context_parts.append(
                        f"  [{i}] {metadata.get('action', '')} - "
                        f"{metadata.get('reasoning', '')} "
                        f"(信心度: {metadata.get('confidence', 0):.2f})"
                    )

            if trades:
                context_parts.append("\n相关历史交易:")
                for i, trade in enumerate(trades, 1):
                    metadata = trade.get("metadata", {})
                    profit = metadata.get('profit_pct', 0)
                    result = "盈利" if profit > 0 else "亏损"
                    context_parts.append(
                        f"  [{i}] {metadata.get('side', '')} {result} {profit:.2f}% - "
                        f"{metadata.get('lessons', '')}"
                    )

            if not context_parts:
                return f"{pair} 暂无相关历史"

            return "\n".join(context_parts)

        except Exception as e:
            logger.error(f"获取相关上下文失败: {e}")
            return "获取历史上下文失败"

    def compress_history(self) -> Dict[str, int]:
        """
        压缩历史数据(删除低质量记录)

        Returns:
            各collection删除的数量
        """
        if not self.enabled:
            return {}

        try:
            deleted_counts = {}

            # 清理决策记录(保留最近的)
            max_decisions = self.config.get("max_history_size", 10000)
            # TODO: 实现基于时间或质量的清理逻辑

            # 清理旧交易记录
            cleanup_days = self.config.get("cleanup_days", 30)
            for col_type, col_name in self.collections.items():
                count = self.vector_store.delete_old_records(col_name, cleanup_days)
                deleted_counts[col_type] = count

            logger.info(f"历史数据压缩完成: {deleted_counts}")
            return deleted_counts

        except Exception as e:
            logger.error(f"压缩历史数据失败: {e}")
            return {}

    def cleanup_low_quality(self) -> int:
        """
        清理低质量记录

        Returns:
            删除数量
        """
        if not self.enabled:
            return 0

        try:
            # 清理低信心度且亏损的交易记录
            # TODO: 实现更复杂的质量评估逻辑

            logger.info("低质量记录清理完成")
            return 0

        except Exception as e:
            logger.error(f"清理低质量记录失败: {e}")
            return 0

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取RAG系统统计信息

        Returns:
            统计信息
        """
        stats = {
            "enabled": self.enabled,
            "collections": {}
        }

        for col_type, col_name in self.collections.items():
            stats["collections"][col_type] = self.vector_store.get_collection_stats(col_name)

        return stats
