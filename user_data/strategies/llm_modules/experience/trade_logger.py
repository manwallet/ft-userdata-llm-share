"""
交易日志模块
记录所有决策和交易到JSONL文件
"""
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class TradeLogger:
    """交易日志记录器"""

    def __init__(self, experience_config: Dict[str, Any]):
        """
        初始化交易日志

        Args:
            experience_config: 经验系统配置
        """
        self.config = experience_config

        # 日志文件路径
        self.decision_log_path = Path(experience_config.get(
            "decision_log_path",
            "./user_data/logs/llm_decisions.jsonl"
        ))
        self.trade_log_path = Path(experience_config.get(
            "trade_log_path",
            "./user_data/logs/trade_experience.jsonl"
        ))

        # 确保目录存在
        self.decision_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.trade_log_path.parent.mkdir(parents=True, exist_ok=True)

        self.log_decisions = experience_config.get("log_decisions", True)
        self.log_trades = experience_config.get("log_trades", True)

        logger.info("交易日志已初始化")

    def log_decision(
        self,
        pair: str,
        action: str,
        decision: str,
        reasoning: str,
        confidence: float,
        market_context: Optional[Dict[str, Any]] = None,
        function_calls: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """
        记录决策

        Args:
            pair: 交易对
            action: 操作类型
            decision: 决策内容
            reasoning: 推理过程
            confidence: 信心度
            market_context: 市场上下文
            function_calls: 使用的函数调用

        Returns:
            是否成功
        """
        if not self.log_decisions:
            return True

        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "pair": pair,
                "action": action,
                "decision": decision,
                "reasoning": reasoning,
                "confidence": confidence,
                "market_context": market_context or {},
                "function_calls": function_calls or []
            }

            self._write_jsonl(self.decision_log_path, log_entry)
            return True

        except Exception as e:
            logger.error(f"记录决策失败: {e}")
            return False

    def log_trade(
        self,
        trade_id: int,
        pair: str,
        side: str,
        entry_time: datetime,
        entry_price: float,
        entry_reason: str,
        exit_time: Optional[datetime] = None,
        exit_price: Optional[float] = None,
        exit_reason: Optional[str] = None,
        profit_pct: Optional[float] = None,
        profit_abs: Optional[float] = None,
        leverage: float = 1,
        stake_amount: float = 0,
        max_drawdown: float = 0,
        lessons: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        记录交易

        Args:
            trade_id: 交易ID
            pair: 交易对
            side: 方向 (long/short)
            entry_time: 入场时间
            entry_price: 入场价格
            entry_reason: 入场理由
            exit_time: 出场时间
            exit_price: 出场价格
            exit_reason: 出场理由
            profit_pct: 盈亏百分比
            profit_abs: 盈亏绝对值
            leverage: 杠杆
            stake_amount: 投入金额
            max_drawdown: 最大回撤
            lessons: 经验教训
            metadata: 额外数据

        Returns:
            是否成功
        """
        if not self.log_trades:
            return True

        try:
            # 计算持仓时长
            duration_minutes = 0
            if exit_time and entry_time:
                duration = exit_time - entry_time
                duration_minutes = int(duration.total_seconds() / 60)

            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "trade_id": trade_id,
                "pair": pair,
                "side": side,
                "entry_time": entry_time.isoformat() if entry_time else None,
                "entry_price": entry_price,
                "entry_reason": entry_reason,
                "exit_time": exit_time.isoformat() if exit_time else None,
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "profit_pct": profit_pct,
                "profit_abs": profit_abs,
                "leverage": leverage,
                "stake_amount": stake_amount,
                "duration_minutes": duration_minutes,
                "max_drawdown": max_drawdown,
                "lessons": lessons,
                "metadata": metadata or {}
            }

            self._write_jsonl(self.trade_log_path, log_entry)
            return True

        except Exception as e:
            logger.error(f"记录交易失败: {e}")
            return False

    def query_logs(
        self,
        log_type: str = "decisions",
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        查询日志

        Args:
            log_type: 日志类型 (decisions/trades)
            filters: 过滤条件
            limit: 最大返回数量

        Returns:
            日志条目列表
        """
        try:
            log_path = self.decision_log_path if log_type == "decisions" else self.trade_log_path

            if not log_path.exists():
                return []

            results = []
            with open(log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if len(results) >= limit:
                        break

                    try:
                        entry = json.loads(line.strip())

                        # 应用过滤器
                        if filters:
                            match = True
                            for key, value in filters.items():
                                if entry.get(key) != value:
                                    match = False
                                    break
                            if not match:
                                continue

                        results.append(entry)

                    except json.JSONDecodeError:
                        continue

            return results[-limit:]  # 返回最新的N条

        except Exception as e:
            logger.error(f"查询日志失败: {e}")
            return []

    def export_summary(
        self,
        output_path: str,
        log_type: str = "trades"
    ) -> bool:
        """
        导出日志摘要

        Args:
            output_path: 输出路径
            log_type: 日志类型

        Returns:
            是否成功
        """
        try:
            logs = self.query_logs(log_type=log_type, limit=10000)

            if log_type == "trades":
                summary = self._generate_trade_summary(logs)
            else:
                summary = self._generate_decision_summary(logs)

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            logger.info(f"摘要已导出到: {output_path}")
            return True

        except Exception as e:
            logger.error(f"导出摘要失败: {e}")
            return False

    def _write_jsonl(self, file_path: Path, data: Dict[str, Any]):
        """写入JSONL文件"""
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

    def _generate_trade_summary(self, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成交易摘要"""
        if not logs:
            return {"total_trades": 0}

        total_trades = len(logs)
        profitable_trades = [log for log in logs if log.get("profit_pct", 0) > 0]
        losing_trades = [log for log in logs if log.get("profit_pct", 0) <= 0]

        total_profit = sum(log.get("profit_pct", 0) for log in logs)
        avg_profit = total_profit / total_trades if total_trades > 0 else 0

        # 按交易对统计
        pair_stats = {}
        for log in logs:
            pair = log.get("pair", "UNKNOWN")
            if pair not in pair_stats:
                pair_stats[pair] = {"count": 0, "profit": 0, "wins": 0}

            pair_stats[pair]["count"] += 1
            pair_stats[pair]["profit"] += log.get("profit_pct", 0)
            if log.get("profit_pct", 0) > 0:
                pair_stats[pair]["wins"] += 1

        return {
            "total_trades": total_trades,
            "profitable_trades": len(profitable_trades),
            "losing_trades": len(losing_trades),
            "win_rate": len(profitable_trades) / total_trades if total_trades > 0 else 0,
            "total_profit_pct": round(total_profit, 2),
            "avg_profit_pct": round(avg_profit, 2),
            "best_trade": max(logs, key=lambda x: x.get("profit_pct", 0)) if logs else None,
            "worst_trade": min(logs, key=lambda x: x.get("profit_pct", 0)) if logs else None,
            "pair_stats": pair_stats
        }

    def _generate_decision_summary(self, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成决策摘要"""
        if not logs:
            return {"total_decisions": 0}

        total_decisions = len(logs)

        # 按操作类型统计
        action_counts = {}
        for log in logs:
            action = log.get("action", "UNKNOWN")
            action_counts[action] = action_counts.get(action, 0) + 1

        # 平均信心度
        avg_confidence = sum(log.get("confidence", 0) for log in logs) / total_decisions if total_decisions > 0 else 0

        return {
            "total_decisions": total_decisions,
            "action_counts": action_counts,
            "avg_confidence": round(avg_confidence, 2)
        }

    def rotate_logs(self, max_size_mb: int = 100):
        """轮转日志文件"""
        for log_path in [self.decision_log_path, self.trade_log_path]:
            if not log_path.exists():
                continue

            size_mb = log_path.stat().st_size / (1024 * 1024)
            if size_mb > max_size_mb:
                # 重命名旧文件
                backup_path = log_path.with_suffix(f".{datetime.now().strftime('%Y%m%d')}.jsonl")
                log_path.rename(backup_path)
                logger.info(f"日志已轮转: {log_path} -> {backup_path}")
