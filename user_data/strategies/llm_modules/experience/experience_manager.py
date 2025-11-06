"""
经验管理器模块
管理交易经验的存储、分析和学习
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


class ExperienceManager:
    """经验学习管理器"""

    def __init__(self, trade_logger, rag_manager=None):
        """
        初始化经验管理器

        Args:
            trade_logger: 交易日志记录器
            rag_manager: RAG管理器(可选)
        """
        self.trade_logger = trade_logger
        self.rag_manager = rag_manager

        # 内存缓存
        self._pair_performance: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._recent_mistakes: List[Dict[str, Any]] = []
        self._success_patterns: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        logger.info("经验管理器已初始化")

    def log_decision_with_context(
        self,
        pair: str,
        action: str,
        decision: str,
        reasoning: str,
        confidence: float,
        market_context: Dict[str, Any],
        function_calls: List[Dict[str, Any]]
    ):
        """
        记录决策并存储到RAG系统

        Args:
            pair: 交易对
            action: 操作类型
            decision: 决策内容
            reasoning: 推理过程
            confidence: 信心度
            market_context: 市场上下文
            function_calls: 函数调用历史
        """
        # 记录到日志文件
        self.trade_logger.log_decision(
            pair=pair,
            action=action,
            decision=decision,
            reasoning=reasoning,
            confidence=confidence,
            market_context=market_context,
            function_calls=function_calls
        )

        # 存储到RAG系统
        if self.rag_manager and self.rag_manager.enabled:
            market_desc = self._format_market_context(market_context)
            self.rag_manager.store_decision(
                pair=pair,
                action=action,
                market_context=market_desc,
                decision=decision,
                reasoning=reasoning,
                confidence=confidence,
                metadata={
                    "function_calls_count": len(function_calls)
                }
            )

    def log_trade_completion(
        self,
        trade_id: int,
        pair: str,
        side: str,
        entry_time: datetime,
        entry_price: float,
        entry_reason: str,
        exit_time: datetime,
        exit_price: float,
        exit_reason: str,
        profit_pct: float,
        profit_abs: float,
        leverage: float,
        stake_amount: float,
        max_drawdown: float,
        market_condition: str
    ):
        """
        记录交易完成

        Args:
            (参数同trade_logger.log_trade)
        """
        # 分析交易并提取教训
        lessons = self._analyze_trade(
            pair=pair,
            side=side,
            profit_pct=profit_pct,
            entry_reason=entry_reason,
            exit_reason=exit_reason,
            max_drawdown=max_drawdown
        )

        # 记录到日志
        self.trade_logger.log_trade(
            trade_id=trade_id,
            pair=pair,
            side=side,
            entry_time=entry_time,
            entry_price=entry_price,
            entry_reason=entry_reason,
            exit_time=exit_time,
            exit_price=exit_price,
            exit_reason=exit_reason,
            profit_pct=profit_pct,
            profit_abs=profit_abs,
            leverage=leverage,
            stake_amount=stake_amount,
            max_drawdown=max_drawdown,
            lessons=lessons
        )

        # 存储到RAG系统
        if self.rag_manager and self.rag_manager.enabled:
            duration_minutes = int((exit_time - entry_time).total_seconds() / 60)

            self.rag_manager.store_trade(
                pair=pair,
                side=side,
                entry_reason=entry_reason,
                exit_reason=exit_reason,
                profit_pct=profit_pct,
                duration_minutes=duration_minutes,
                entry_price=entry_price,
                exit_price=exit_price,
                leverage=leverage,
                market_condition=market_condition,
                lessons=lessons,
                metadata={
                    "trade_id": trade_id,
                    "max_drawdown": max_drawdown
                }
            )

        # 更新内存统计
        self._update_pair_performance(pair, profit_pct, side)

        # 记录错误(如果是亏损交易)
        if profit_pct < -2.0:  # 亏损超过2%
            self._record_mistake(
                pair=pair,
                side=side,
                loss_pct=profit_pct,
                reason=entry_reason,
                lessons=lessons
            )

        # 记录成功模式(如果是盈利交易)
        if profit_pct > 2.0:  # 盈利超过2%
            self._record_success_pattern(
                pair=pair,
                side=side,
                profit_pct=profit_pct,
                entry_reason=entry_reason,
                market_condition=market_condition
            )

    def get_pair_statistics(self, pair: str) -> Dict[str, Any]:
        """
        获取交易对统计

        Args:
            pair: 交易对

        Returns:
            统计信息
        """
        # 从日志查询
        trade_logs = self.trade_logger.query_logs(
            log_type="trades",
            filters={"pair": pair},
            limit=1000
        )

        if not trade_logs:
            return {
                "pair": pair,
                "total_trades": 0,
                "message": "暂无交易记录"
            }

        total_trades = len(trade_logs)
        wins = sum(1 for log in trade_logs if log.get("profit_pct", 0) > 0)
        losses = total_trades - wins

        total_profit = sum(log.get("profit_pct", 0) for log in trade_logs)
        avg_profit = total_profit / total_trades

        avg_win = sum(log.get("profit_pct", 0) for log in trade_logs if log.get("profit_pct", 0) > 0) / wins if wins > 0 else 0
        avg_loss = sum(log.get("profit_pct", 0) for log in trade_logs if log.get("profit_pct", 0) < 0) / losses if losses > 0 else 0

        return {
            "pair": pair,
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": wins / total_trades if total_trades > 0 else 0,
            "total_profit_pct": round(total_profit, 2),
            "avg_profit_pct": round(avg_profit, 2),
            "avg_win_pct": round(avg_win, 2),
            "avg_loss_pct": round(avg_loss, 2),
            "profit_factor": abs(avg_win * wins / (avg_loss * losses)) if losses > 0 and avg_loss != 0 else 0
        }

    def analyze_mistakes(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        分析最近的错误

        Args:
            limit: 返回数量

        Returns:
            错误分析列表
        """
        return self._recent_mistakes[-limit:]

    def extract_lessons(self, pair: str) -> str:
        """
        提取交易对的经验教训

        Args:
            pair: 交易对

        Returns:
            经验教训文本
        """
        stats = self.get_pair_statistics(pair)

        if stats.get("total_trades", 0) == 0:
            return f"{pair} 暂无交易经验"

        lessons = []

        # 胜率分析
        win_rate = stats.get("win_rate", 0)
        if win_rate > 0.6:
            lessons.append(f"✓ 胜率较高 ({win_rate:.1%})，策略有效")
        elif win_rate < 0.4:
            lessons.append(f"✗ 胜率较低 ({win_rate:.1%})，需要调整策略")

        # 盈亏比分析
        profit_factor = stats.get("profit_factor", 0)
        if profit_factor > 1.5:
            lessons.append(f"✓ 盈亏比良好 ({profit_factor:.2f})")
        elif profit_factor < 1.0:
            lessons.append(f"✗ 盈亏比不佳 ({profit_factor:.2f})，止盈过早或止损过晚")

        # 平均盈利分析
        avg_profit = stats.get("avg_profit_pct", 0)
        if avg_profit > 0:
            lessons.append(f"✓ 整体盈利 (平均 {avg_profit:.2f}%)")
        else:
            lessons.append(f"✗ 整体亏损 (平均 {avg_profit:.2f}%)")

        return "\n".join(lessons)

    def _analyze_trade(
        self,
        pair: str,
        side: str,
        profit_pct: float,
        entry_reason: str,
        exit_reason: str,
        max_drawdown: float
    ) -> str:
        """分析单笔交易并提取教训"""
        lessons = []

        if profit_pct > 0:
            lessons.append(f"成功 {side} {profit_pct:.2f}%")
            if profit_pct > 10:
                lessons.append("大幅盈利，策略非常有效")
        else:
            lessons.append(f"亏损 {side} {profit_pct:.2f}%")
            if abs(max_drawdown) > abs(profit_pct) * 1.5:
                lessons.append("回撤过大，可能入场时机不佳或止损设置不当")

        # 可以添加更多分析逻辑...

        return " | ".join(lessons)

    def _update_pair_performance(self, pair: str, profit_pct: float, side: str):
        """更新交易对表现统计"""
        if pair not in self._pair_performance:
            self._pair_performance[pair] = {
                "total_trades": 0,
                "total_profit": 0,
                "long_count": 0,
                "short_count": 0
            }

        self._pair_performance[pair]["total_trades"] += 1
        self._pair_performance[pair]["total_profit"] += profit_pct

        if side == "long":
            self._pair_performance[pair]["long_count"] += 1
        else:
            self._pair_performance[pair]["short_count"] += 1

    def _record_mistake(
        self,
        pair: str,
        side: str,
        loss_pct: float,
        reason: str,
        lessons: str
    ):
        """记录错误"""
        mistake = {
            "timestamp": datetime.now().isoformat(),
            "pair": pair,
            "side": side,
            "loss_pct": loss_pct,
            "reason": reason,
            "lessons": lessons
        }

        self._recent_mistakes.append(mistake)

        # 只保留最近100个错误
        if len(self._recent_mistakes) > 100:
            self._recent_mistakes = self._recent_mistakes[-100:]

    def _record_success_pattern(
        self,
        pair: str,
        side: str,
        profit_pct: float,
        entry_reason: str,
        market_condition: str
    ):
        """记录成功模式"""
        pattern = {
            "timestamp": datetime.now().isoformat(),
            "side": side,
            "profit_pct": profit_pct,
            "entry_reason": entry_reason,
            "market_condition": market_condition
        }

        self._success_patterns[pair].append(pattern)

        # 只保留最近50个成功案例
        if len(self._success_patterns[pair]) > 50:
            self._success_patterns[pair] = self._success_patterns[pair][-50:]

    def _format_market_context(self, context: Dict[str, Any]) -> str:
        """格式化市场上下文为文本"""
        parts = []

        for key, value in context.items():
            if isinstance(value, (int, float)):
                parts.append(f"{key}: {value:.4f}")
            else:
                parts.append(f"{key}: {value}")

        return ", ".join(parts)

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "tracked_pairs": len(self._pair_performance),
            "recent_mistakes": len(self._recent_mistakes),
            "success_patterns": sum(len(patterns) for patterns in self._success_patterns.values())
        }
