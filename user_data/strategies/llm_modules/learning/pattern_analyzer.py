"""
模式分析器
从历史交易中提取高胜率条件、高风险条件、常见错误模式
"""
import logging
import re
from typing import List, Dict, Any, Tuple
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


class PatternAnalyzer:
    """交易模式分析器"""

    def __init__(self, min_sample_size: int = 5):
        """
        初始化模式分析器

        Args:
            min_sample_size: 最小样本数量（少于此数量不做统计）
        """
        self.min_sample_size = min_sample_size

    def analyze_patterns(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析交易模式

        Args:
            trades: 交易列表

        Returns:
            模式分析结果
        """
        if len(trades) < self.min_sample_size:
            return {
                'total_trades': len(trades),
                'message': f'样本量不足（需要至少{self.min_sample_size}笔交易）'
            }

        return {
            'high_win_conditions': self._find_high_win_conditions(trades),
            'high_risk_conditions': self._find_high_risk_conditions(trades),
            'common_mistakes': self._extract_common_mistakes(trades),
            'time_patterns': self._analyze_time_patterns(trades),
            'side_analysis': self._analyze_side_performance(trades)
        }

    def _find_high_win_conditions(self, trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        识别高胜率条件

        Args:
            trades: 交易列表

        Returns:
            高胜率条件列表
        """
        conditions = []

        # 按方向统计
        long_trades = [t for t in trades if t.get('side') == 'long']
        short_trades = [t for t in trades if t.get('side') == 'short']

        if len(long_trades) >= self.min_sample_size:
            long_wins = sum(1 for t in long_trades if t.get('profit_pct', 0) > 0)
            long_win_rate = long_wins / len(long_trades)
            if long_win_rate >= 0.6:  # 胜率≥60%
                conditions.append({
                    'condition': '做多',
                    'win_rate': long_win_rate,
                    'sample_size': len(long_trades),
                    'wins': long_wins,
                    'avg_profit': sum(t.get('profit_pct', 0) for t in long_trades) / len(long_trades)
                })

        if len(short_trades) >= self.min_sample_size:
            short_wins = sum(1 for t in short_trades if t.get('profit_pct', 0) > 0)
            short_win_rate = short_wins / len(short_trades)
            if short_win_rate >= 0.6:
                conditions.append({
                    'condition': '做空',
                    'win_rate': short_win_rate,
                    'sample_size': len(short_trades),
                    'wins': short_wins,
                    'avg_profit': sum(t.get('profit_pct', 0) for t in short_trades) / len(short_trades)
                })

        # 按入场理由关键词分析
        entry_keywords = self._extract_entry_keywords(trades)
        for keyword, keyword_trades in entry_keywords.items():
            if len(keyword_trades) >= self.min_sample_size:
                wins = sum(1 for t in keyword_trades if t.get('profit_pct', 0) > 0)
                win_rate = wins / len(keyword_trades)
                if win_rate >= 0.65:  # 更高的胜率阈值
                    conditions.append({
                        'condition': f'入场信号包含"{keyword}"',
                        'win_rate': win_rate,
                        'sample_size': len(keyword_trades),
                        'wins': wins,
                        'avg_profit': sum(t.get('profit_pct', 0) for t in keyword_trades) / len(keyword_trades)
                    })

        # 按胜率排序
        conditions = sorted(conditions, key=lambda x: x['win_rate'], reverse=True)

        return conditions[:5]  # 返回前5个

    def _find_high_risk_conditions(self, trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        识别高风险条件

        Args:
            trades: 交易列表

        Returns:
            高风险条件列表
        """
        risks = []

        # 按方向统计亏损
        long_trades = [t for t in trades if t.get('side') == 'long']
        short_trades = [t for t in trades if t.get('side') == 'short']

        if len(long_trades) >= self.min_sample_size:
            long_losses = sum(1 for t in long_trades if t.get('profit_pct', 0) < 0)
            long_loss_rate = long_losses / len(long_trades)
            if long_loss_rate >= 0.6:  # 亏损率≥60%
                risks.append({
                    'condition': '做多',
                    'loss_rate': long_loss_rate,
                    'sample_size': len(long_trades),
                    'losses': long_losses,
                    'avg_loss': sum(t.get('profit_pct', 0) for t in long_trades if t.get('profit_pct', 0) < 0) / long_losses if long_losses > 0 else 0
                })

        if len(short_trades) >= self.min_sample_size:
            short_losses = sum(1 for t in short_trades if t.get('profit_pct', 0) < 0)
            short_loss_rate = short_losses / len(short_trades)
            if short_loss_rate >= 0.6:
                risks.append({
                    'condition': '做空',
                    'loss_rate': short_loss_rate,
                    'sample_size': len(short_trades),
                    'losses': short_losses,
                    'avg_loss': sum(t.get('profit_pct', 0) for t in short_trades if t.get('profit_pct', 0) < 0) / short_losses if short_losses > 0 else 0
                })

        # 按时段分析（简单版：判断是否有夜间交易亏损多的模式）
        night_trades = []
        for t in trades:
            entry_time = t.get('entry_time', '')
            if entry_time:
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(entry_time)
                    hour = dt.hour
                    if 22 <= hour or hour <= 2:  # 22:00-02:00
                        night_trades.append(t)
                except:
                    pass

        if len(night_trades) >= self.min_sample_size:
            night_losses = sum(1 for t in night_trades if t.get('profit_pct', 0) < 0)
            night_loss_rate = night_losses / len(night_trades)
            if night_loss_rate >= 0.6:
                risks.append({
                    'condition': '夜间交易（22:00-02:00）',
                    'loss_rate': night_loss_rate,
                    'sample_size': len(night_trades),
                    'losses': night_losses,
                    'avg_loss': sum(t.get('profit_pct', 0) for t in night_trades if t.get('profit_pct', 0) < 0) / night_losses if night_losses > 0 else 0
                })

        # 按平仓理由关键词分析
        exit_keywords = self._extract_exit_keywords(trades)
        for keyword, keyword_trades in exit_keywords.items():
            if len(keyword_trades) >= self.min_sample_size:
                losses = sum(1 for t in keyword_trades if t.get('profit_pct', 0) < 0)
                loss_rate = losses / len(keyword_trades)
                if loss_rate >= 0.7:  # 高亏损率阈值
                    risks.append({
                        'condition': f'出场理由包含"{keyword}"',
                        'loss_rate': loss_rate,
                        'sample_size': len(keyword_trades),
                        'losses': losses,
                        'avg_loss': sum(t.get('profit_pct', 0) for t in keyword_trades if t.get('profit_pct', 0) < 0) / losses if losses > 0 else 0
                    })

        # 按亏损率排序
        risks = sorted(risks, key=lambda x: x['loss_rate'], reverse=True)

        return risks[:5]  # 返回前5个

    def _extract_common_mistakes(self, trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        提取常见错误模式

        Args:
            trades: 交易列表

        Returns:
            常见错误列表
        """
        mistakes = []

        # 只分析亏损交易
        losing_trades = [t for t in trades if t.get('profit_pct', 0) < -2.0]  # 亏损>2%

        if len(losing_trades) < self.min_sample_size:
            return []

        # 1. 分析出场理由中的关键词
        exit_reasons = Counter()
        for t in losing_trades:
            reason = t.get('exit_reason', '')
            # 提取关键词
            keywords = self._extract_keywords_from_reason(reason)
            for kw in keywords:
                exit_reasons[kw] += 1

        # 找出频繁出现的出场理由
        for keyword, count in exit_reasons.most_common(5):
            if count >= 3:  # 至少出现3次
                related_trades = [t for t in losing_trades if keyword in t.get('exit_reason', '')]
                avg_loss = sum(t.get('profit_pct', 0) for t in related_trades) / len(related_trades)
                mistakes.append({
                    'mistake_type': '出场问题',
                    'description': f'"{keyword}" 导致亏损',
                    'frequency': count,
                    'avg_loss': avg_loss,
                    'sample_size': len(related_trades)
                })

        # 2. 分析入场理由中的问题模式
        entry_reasons = Counter()
        for t in losing_trades:
            reason = t.get('entry_reason', '')
            keywords = self._extract_keywords_from_reason(reason)
            for kw in keywords:
                entry_reasons[kw] += 1

        for keyword, count in entry_reasons.most_common(5):
            if count >= 3:
                related_trades = [t for t in losing_trades if keyword in t.get('entry_reason', '')]
                avg_loss = sum(t.get('profit_pct', 0) for t in related_trades) / len(related_trades)
                mistakes.append({
                    'mistake_type': '入场问题',
                    'description': f'"{keyword}" 信号不可靠',
                    'frequency': count,
                    'avg_loss': avg_loss,
                    'sample_size': len(related_trades)
                })

        # 3. 持仓时间过长/过短
        # TODO: 分析持仓时长与盈亏的关系

        return mistakes[:5]

    def _analyze_time_patterns(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析时间模式

        Args:
            trades: 交易列表

        Returns:
            时间模式统计
        """
        from datetime import datetime

        hour_performance = defaultdict(list)

        for t in trades:
            entry_time = t.get('entry_time', '')
            if not entry_time:
                continue

            try:
                dt = datetime.fromisoformat(entry_time)
                hour = dt.hour
                profit = t.get('profit_pct', 0)
                hour_performance[hour].append(profit)
            except:
                continue

        # 计算每小时平均盈亏
        hour_stats = {}
        for hour, profits in hour_performance.items():
            if len(profits) >= 3:  # 至少3笔交易
                hour_stats[hour] = {
                    'avg_profit': sum(profits) / len(profits),
                    'trade_count': len(profits),
                    'win_rate': sum(1 for p in profits if p > 0) / len(profits)
                }

        # 找出最佳和最差时段
        if hour_stats:
            best_hour = max(hour_stats.items(), key=lambda x: x[1]['avg_profit'])
            worst_hour = min(hour_stats.items(), key=lambda x: x[1]['avg_profit'])

            return {
                'best_hour': {
                    'hour': best_hour[0],
                    'avg_profit': best_hour[1]['avg_profit'],
                    'trade_count': best_hour[1]['trade_count']
                },
                'worst_hour': {
                    'hour': worst_hour[0],
                    'avg_profit': worst_hour[1]['avg_profit'],
                    'trade_count': worst_hour[1]['trade_count']
                }
            }

        return {}

    def _analyze_side_performance(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析做多/做空表现

        Args:
            trades: 交易列表

        Returns:
            方向统计
        """
        long_trades = [t for t in trades if t.get('side') == 'long']
        short_trades = [t for t in trades if t.get('side') == 'short']

        def calc_stats(side_trades):
            if not side_trades:
                return None
            wins = sum(1 for t in side_trades if t.get('profit_pct', 0) > 0)
            total_profit = sum(t.get('profit_pct', 0) for t in side_trades)
            return {
                'trade_count': len(side_trades),
                'wins': wins,
                'losses': len(side_trades) - wins,
                'win_rate': wins / len(side_trades),
                'total_profit': total_profit,
                'avg_profit': total_profit / len(side_trades)
            }

        return {
            'long': calc_stats(long_trades),
            'short': calc_stats(short_trades)
        }

    def _extract_entry_keywords(self, trades: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
        """提取入场理由中的关键词"""
        keywords = defaultdict(list)
        common_words = ['超卖', '金叉', '突破', '支撑', '反弹', '趋势', '强势', '回调', '底部']

        for t in trades:
            reason = t.get('entry_reason', '')
            for word in common_words:
                if word in reason:
                    keywords[word].append(t)

        return keywords

    def _extract_exit_keywords(self, trades: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
        """提取出场理由中的关键词"""
        keywords = defaultdict(list)
        common_words = ['止损', '止盈', '反转', '回落', '阻力', '破位', '信号', '超买']

        for t in trades:
            reason = t.get('exit_reason', '')
            for word in common_words:
                if word in reason:
                    keywords[word].append(t)

        return keywords

    def _extract_keywords_from_reason(self, reason: str) -> List[str]:
        """从理由文本中提取关键词"""
        keywords = []
        common_patterns = [
            '止损', '止盈', '反转', '突破', '跌破', '超买', '超卖',
            '金叉', '死叉', '支撑', '阻力', '回调', '反弹', '趋势'
        ]

        for pattern in common_patterns:
            if pattern in reason:
                keywords.append(pattern)

        return keywords

    def format_patterns_for_context(
        self,
        pair: str,
        trades: List[Dict[str, Any]]
    ) -> str:
        """
        格式化模式分析结果为上下文文本

        Args:
            pair: 交易对
            trades: 交易列表

        Returns:
            格式化的模式分析文本
        """
        patterns = self.analyze_patterns(trades)

        if 'message' in patterns:
            return f"【{pair} 模式分析】{patterns['message']}"

        lines = [f"【{pair} 经验规则】"]

        # 高胜率条件
        if patterns.get('high_win_conditions'):
            lines.append("\n✓ 高胜率条件：")
            for cond in patterns['high_win_conditions'][:3]:
                lines.append(
                    f"  - {cond['condition']}: 胜率 {cond['win_rate']:.1%} "
                    f"({cond['wins']}胜/{cond['sample_size']}笔) "
                    f"平均盈利 {cond['avg_profit']:+.2f}%"
                )

        # 高风险条件
        if patterns.get('high_risk_conditions'):
            lines.append("\n✗ 高风险条件：")
            for risk in patterns['high_risk_conditions'][:3]:
                lines.append(
                    f"  - {risk['condition']}: 亏损率 {risk['loss_rate']:.1%} "
                    f"({risk['losses']}亏/{risk['sample_size']}笔) "
                    f"平均亏损 {risk['avg_loss']:.2f}%"
                )

        # 常见错误
        if patterns.get('common_mistakes'):
            lines.append("\n⚠️  常见错误：")
            for mistake in patterns['common_mistakes'][:3]:
                lines.append(
                    f"  {mistake['frequency']}次 {mistake['mistake_type']}: "
                    f"{mistake['description']} (平均亏损 {mistake['avg_loss']:.2f}%)"
                )

        return '\n'.join(lines)
