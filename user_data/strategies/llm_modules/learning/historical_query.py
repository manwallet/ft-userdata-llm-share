"""
历史查询引擎
从trade_experience.jsonl查询相关历史交易，提供给LLM参考
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


class HistoricalQueryEngine:
    """历史交易查询引擎"""

    def __init__(self, trade_log_path: str):
        """
        初始化查询引擎

        Args:
            trade_log_path: trade_experience.jsonl 文件路径
        """
        self.trade_log_path = Path(trade_log_path)
        self._cache = []  # 缓存所有交易
        self._last_load_time = None
        self._load_trades()

    def _load_trades(self):
        """从JSONL文件加载所有交易"""
        if not self.trade_log_path.exists():
            logger.warning(f"交易日志文件不存在: {self.trade_log_path}")
            self._cache = []
            return

        try:
            self._cache = []
            with open(self.trade_log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        trade = json.loads(line)
                        self._cache.append(trade)

            self._last_load_time = datetime.now()
            logger.debug(f"已加载 {len(self._cache)} 笔历史交易")
        except Exception as e:
            logger.error(f"加载交易日志失败: {e}")
            self._cache = []

    def reload_if_needed(self, max_age_seconds: int = 60):
        """如果缓存过期则重新加载"""
        if not self._last_load_time or \
           (datetime.now() - self._last_load_time).total_seconds() > max_age_seconds:
            self._load_trades()

    def query_recent_trades(
        self,
        pair: Optional[str] = None,
        limit: int = 10,
        days: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        查询最近的交易

        Args:
            pair: 交易对（可选，不指定则返回所有交易对）
            limit: 返回数量
            days: 最近N天（可选）

        Returns:
            交易列表（按时间倒序）
        """
        self.reload_if_needed()

        trades = self._cache

        # 按交易对筛选
        if pair:
            trades = [t for t in trades if t.get('pair') == pair]

        # 按时间筛选
        if days:
            cutoff = datetime.now() - timedelta(days=days)
            trades = [
                t for t in trades
                if datetime.fromisoformat(t.get('exit_time', '')) > cutoff
            ]

        # 按时间倒序排序
        trades = sorted(
            trades,
            key=lambda x: x.get('exit_time', ''),
            reverse=True
        )

        return trades[:limit]

    def query_similar_conditions(
        self,
        pair: str,
        current_rsi: Optional[float] = None,
        current_trend: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        查询相似市场条件下的历史交易

        Args:
            pair: 交易对
            current_rsi: 当前RSI值（可选）
            current_trend: 当前趋势（可选）
            limit: 返回数量

        Returns:
            相似条件的交易列表
        """
        self.reload_if_needed()

        trades = [t for t in self._cache if t.get('pair') == pair]

        # RSI相似度筛选（±10范围内）
        if current_rsi is not None:
            similar_trades = []
            for t in trades:
                # 从market_condition中提取RSI（如果有）
                market_condition = t.get('market_condition', '')
                # 简单解析，后续可以改进
                # TODO: 更好的方式是在log_trade时单独存储市场指标
                similar_trades.append(t)
            trades = similar_trades

        # 趋势筛选
        if current_trend:
            # TODO: 添加趋势匹配逻辑
            pass

        # 按时间倒序
        trades = sorted(
            trades,
            key=lambda x: x.get('exit_time', ''),
            reverse=True
        )

        return trades[:limit]

    def get_pair_summary(self, pair: str, days: int = 30) -> Dict[str, Any]:
        """
        获取交易对统计摘要

        Args:
            pair: 交易对
            days: 统计天数

        Returns:
            统计信息字典
        """
        trades = self.query_recent_trades(pair=pair, limit=1000, days=days)

        if not trades:
            return {
                'pair': pair,
                'total_trades': 0,
                'message': '暂无历史交易'
            }

        total_trades = len(trades)
        wins = sum(1 for t in trades if t.get('profit_pct', 0) > 0)
        losses = total_trades - wins

        total_profit = sum(t.get('profit_pct', 0) for t in trades)
        avg_profit = total_profit / total_trades if total_trades > 0 else 0

        win_profits = [t.get('profit_pct', 0) for t in trades if t.get('profit_pct', 0) > 0]
        loss_profits = [t.get('profit_pct', 0) for t in trades if t.get('profit_pct', 0) < 0]

        avg_win = sum(win_profits) / len(win_profits) if win_profits else 0
        avg_loss = sum(loss_profits) / len(loss_profits) if loss_profits else 0

        # 按方向统计
        long_trades = [t for t in trades if t.get('side') == 'long']
        short_trades = [t for t in trades if t.get('side') == 'short']

        return {
            'pair': pair,
            'period_days': days,
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'win_rate': wins / total_trades if total_trades > 0 else 0,
            'total_profit_pct': round(total_profit, 2),
            'avg_profit_pct': round(avg_profit, 2),
            'avg_win_pct': round(avg_win, 2),
            'avg_loss_pct': round(avg_loss, 2),
            'profit_factor': abs(avg_win * wins / (avg_loss * losses)) if losses > 0 and avg_loss != 0 else 0,
            'long_trades': len(long_trades),
            'short_trades': len(short_trades),
            'recent_5_trades': trades[:5]
        }

    def format_recent_trades_for_context(
        self,
        pair: str,
        limit: int = 5
    ) -> str:
        """
        格式化最近的交易为上下文文本

        Args:
            pair: 交易对
            limit: 返回数量

        Returns:
            格式化的历史经验文本
        """
        from datetime import datetime

        trades = self.query_recent_trades(pair=pair, limit=limit)

        if not trades:
            return f"【{pair} 历史经验】\n暂无历史交易记录"

        lines = [f"【{pair} 最近{len(trades)}笔交易】"]

        for i, trade in enumerate(trades, 1):
            side = trade.get('side', 'unknown')
            profit_pct = trade.get('profit_pct', 0)
            result = '✓' if profit_pct > 0 else '✗'

            # 价格信息
            entry_price = trade.get('entry_price', 0)
            exit_price = trade.get('exit_price', 0)
            leverage = trade.get('leverage', 1)

            # 时间信息
            exit_time = trade.get('exit_time', '')
            time_ago = ''
            if exit_time:
                try:
                    exit_dt = datetime.fromisoformat(exit_time)
                    time_diff = datetime.now() - exit_dt
                    hours_ago = int(time_diff.total_seconds() / 3600)
                    if hours_ago < 1:
                        minutes_ago = int(time_diff.total_seconds() / 60)
                        time_ago = f"{minutes_ago}分钟前"
                    elif hours_ago < 24:
                        time_ago = f"{hours_ago}小时前"
                    else:
                        days_ago = int(hours_ago / 24)
                        time_ago = f"{days_ago}天前"
                except:
                    pass

            # 持仓时长
            duration_minutes = trade.get('duration_minutes', 0)
            if duration_minutes > 0:
                if duration_minutes < 60:
                    duration = f"{duration_minutes}分钟"
                else:
                    hours = duration_minutes / 60
                    duration = f"{hours:.1f}小时"
            else:
                duration = "未知"

            entry_reason = trade.get('entry_reason', '未记录')
            exit_reason = trade.get('exit_reason', '未记录')
            lessons = trade.get('lessons', '')

            # 第一行：基本信息
            lines.append(
                f"{i}. {side} {result} {profit_pct:+.1f}% | "
                f"{time_ago} | 杠杆{leverage}x | 持仓{duration}"
            )

            # 第二行：价格信息
            lines.append(
                f"   入场价 {entry_price:.2f} → 出场价 {exit_price:.2f}"
            )

            # 第三行：入场理由（截取关键部分）
            lines.append(
                f"   入场: {entry_reason[:80]}..."
            )

            # 第四行：出场理由（截取关键部分）
            lines.append(
                f"   出场: {exit_reason[:80]}..."
            )

            # 添加教训（如果有）
            if lessons:
                lines.append(f"   教训: {lessons[:100]}...")

        return '\n'.join(lines)

    def format_pair_summary_for_context(self, pair: str, days: int = 30) -> str:
        """
        格式化交易对统计为上下文文本

        Args:
            pair: 交易对
            days: 统计天数

        Returns:
            格式化的统计文本
        """
        summary = self.get_pair_summary(pair, days)

        if summary['total_trades'] == 0:
            return f"【{pair} 统计】暂无历史数据"

        lines = [
            f"【{pair} 近{days}天统计】",
            f"总交易: {summary['total_trades']}笔 | "
            f"胜率: {summary['win_rate']:.1%} ({summary['wins']}胜/{summary['losses']}负)",
            f"总盈亏: {summary['total_profit_pct']:+.2f}% | "
            f"平均: {summary['avg_profit_pct']:+.2f}%",
            f"平均盈利: +{summary['avg_win_pct']:.2f}% | "
            f"平均亏损: {summary['avg_loss_pct']:.2f}%",
            f"盈亏比: {summary['profit_factor']:.2f} | "
            f"做多{summary['long_trades']}笔 做空{summary['short_trades']}笔"
        ]

        return '\n'.join(lines)
