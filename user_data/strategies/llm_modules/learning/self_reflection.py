"""
自我反思引擎
在交易平仓后对比开仓判断vs实际结果，生成反思报告
"""
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class SelfReflectionEngine:
    """自我反思引擎"""

    def __init__(self):
        """初始化反思引擎"""
        pass

    def generate_reflection(
        self,
        trade_id: int,
        pair: str,
        side: str,
        entry_price: float,
        exit_price: float,
        entry_reason: str,
        exit_reason: str,
        profit_pct: float,
        duration_minutes: int,
        leverage: float,
        position_metrics: Optional[Dict[str, Any]] = None,
        market_changes: Optional[Dict[str, Any]] = None,
        model_score: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        生成交易反思报告

        Args:
            trade_id: 交易ID
            pair: 交易对
            side: 方向 (long/short)
            entry_price: 入场价
            exit_price: 出场价
            entry_reason: 入场理由
            exit_reason: 出场理由
            profit_pct: 盈亏百分比
            duration_minutes: 持仓时长（分钟）
            leverage: 杠杆
            position_metrics: 持仓指标
            market_changes: 市场变化
            model_score: 模型自我评分

        Returns:
            反思报告字典
        """
        # 基础信息
        reflection = {
            'trade_id': trade_id,
            'pair': pair,
            'side': side,
            'profit_pct': profit_pct,
            'result_type': self._classify_result(profit_pct),
            'entry_judgment': entry_reason,
            'actual_outcome': exit_reason,
            'lessons': []
        }

        # 分析入场判断是否正确
        entry_analysis = self._analyze_entry_quality(
            profit_pct=profit_pct,
            entry_reason=entry_reason,
            position_metrics=position_metrics,
            market_changes=market_changes
        )
        reflection['entry_quality'] = entry_analysis

        # 分析出场时机是否合理
        exit_analysis = self._analyze_exit_quality(
            profit_pct=profit_pct,
            exit_reason=exit_reason,
            position_metrics=position_metrics,
            duration_minutes=duration_minutes
        )
        reflection['exit_quality'] = exit_analysis

        # 提取教训
        lessons = self._extract_lessons(
            profit_pct=profit_pct,
            entry_reason=entry_reason,
            exit_reason=exit_reason,
            position_metrics=position_metrics,
            market_changes=market_changes,
            entry_analysis=entry_analysis,
            exit_analysis=exit_analysis
        )
        reflection['lessons'] = lessons

        # 生成总结
        reflection['summary'] = self._generate_summary(
            profit_pct=profit_pct,
            entry_analysis=entry_analysis,
            exit_analysis=exit_analysis,
            lessons=lessons
        )

        return reflection

    def _classify_result(self, profit_pct: float) -> str:
        """分类交易结果"""
        if profit_pct > 5:
            return '大幅盈利'
        elif profit_pct > 2:
            return '盈利'
        elif profit_pct > 0:
            return '小幅盈利'
        elif profit_pct > -2:
            return '小幅亏损'
        elif profit_pct > -5:
            return '亏损'
        else:
            return '大幅亏损'

    def _analyze_entry_quality(
        self,
        profit_pct: float,
        entry_reason: str,
        position_metrics: Optional[Dict[str, Any]],
        market_changes: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        分析入场质量

        Returns:
            入场分析结果
        """
        analysis = {
            'is_correct': profit_pct > 0,
            'score': 0,  # 0-100分
            'strengths': [],
            'weaknesses': []
        }

        # 1. 如果盈利，入场判断基本正确
        if profit_pct > 0:
            analysis['strengths'].append('入场方向正确')
            analysis['score'] += 40

            # 如果是大幅盈利，入场时机也很好
            if profit_pct > 5:
                analysis['strengths'].append('入场时机优秀')
                analysis['score'] += 20
        else:
            analysis['weaknesses'].append('入场方向错误')

        # 2. 检查position_metrics中的max_profit_pct
        if position_metrics:
            max_profit_pct = position_metrics.get('max_profit_pct', 0)
            max_loss_pct = position_metrics.get('max_loss_pct', 0)

            # 如果曾经有过盈利但最终亏损，说明入场可能是对的但持仓管理有问题
            if max_profit_pct > 2 and profit_pct < 0:
                analysis['strengths'].append(f'曾达到最大盈利 {max_profit_pct:+.2f}%')
                analysis['weaknesses'].append('但最终未能止盈，持仓管理问题')
                analysis['score'] += 20  # 入场判断部分正确

            # 如果一直没有盈利过，入场可能确实有问题
            if max_profit_pct <= 0 and profit_pct < 0:
                analysis['weaknesses'].append('开仓后始终未盈利，入场时机不佳')

        # 3. 检查入场理由的关键词
        good_keywords = ['突破', '支撑', '反弹', '超卖', '金叉', '趋势']
        bad_keywords = ['追高', '冲动', '试探', '不确定']

        for keyword in good_keywords:
            if keyword in entry_reason:
                analysis['score'] += 5

        for keyword in bad_keywords:
            if keyword in entry_reason:
                analysis['weaknesses'].append(f'入场理由包含"{keyword}"信号')
                analysis['score'] -= 10

        # 确保分数在0-100之间
        analysis['score'] = max(0, min(100, analysis['score']))

        return analysis

    def _analyze_exit_quality(
        self,
        profit_pct: float,
        exit_reason: str,
        position_metrics: Optional[Dict[str, Any]],
        duration_minutes: int
    ) -> Dict[str, Any]:
        """
        分析出场质量

        Returns:
            出场分析结果
        """
        analysis = {
            'is_timely': False,
            'score': 0,  # 0-100分
            'strengths': [],
            'weaknesses': []
        }

        if not position_metrics:
            return analysis

        max_profit_pct = position_metrics.get('max_profit_pct', 0)
        max_loss_pct = position_metrics.get('max_loss_pct', 0)

        # 1. 分析止盈时机
        if profit_pct > 0:
            profit_retention = (profit_pct / max_profit_pct * 100) if max_profit_pct > 0 else 100

            if profit_retention >= 80:  # 保留了80%以上的利润
                analysis['strengths'].append(f'止盈及时，保留 {profit_retention:.0f}% 利润')
                analysis['score'] += 50
                analysis['is_timely'] = True
            elif profit_retention >= 50:
                analysis['strengths'].append(f'止盈尚可，保留 {profit_retention:.0f}% 利润')
                analysis['score'] += 30
            else:
                analysis['weaknesses'].append(f'止盈过晚，利润回吐 {100-profit_retention:.0f}%')
                analysis['score'] += 10

        # 2. 分析止损时机
        else:  # profit_pct < 0
            if max_profit_pct > 2:  # 曾经盈利超过2%
                analysis['weaknesses'].append(f'曾盈利 {max_profit_pct:+.2f}% 但未止盈，最终亏损')
                analysis['score'] -= 20
            else:
                # 检查止损是否及时
                loss_magnitude = abs(profit_pct)
                if loss_magnitude <= 3:  # 亏损控制在3%以内
                    analysis['strengths'].append('止损及时，控制了亏损')
                    analysis['score'] += 30
                    analysis['is_timely'] = True
                elif loss_magnitude <= 5:
                    analysis['score'] += 10
                else:
                    analysis['weaknesses'].append(f'止损过晚，亏损 {loss_magnitude:.2f}%')

        # 3. 分析持仓时长
        if duration_minutes < 30:
            analysis['weaknesses'].append('持仓时间过短，可能过早出场')
        elif duration_minutes > 1440:  # 超过24小时
            if profit_pct < 2:  # 持仓很久但收益不高
                analysis['weaknesses'].append('持仓时间过长，资金利用率低')

        # 4. 检查出场理由
        good_exit_keywords = ['止盈', '达到目标', '阻力', '信号反转']
        bad_exit_keywords = ['恐慌', '冲动', '不确定']

        for keyword in good_exit_keywords:
            if keyword in exit_reason:
                analysis['score'] += 5

        for keyword in bad_exit_keywords:
            if keyword in exit_reason:
                analysis['weaknesses'].append(f'出场理由"{keyword}"显示情绪化')
                analysis['score'] -= 10

        # 确保分数在0-100之间
        analysis['score'] = max(0, min(100, analysis['score']))

        return analysis

    def _extract_lessons(
        self,
        profit_pct: float,
        entry_reason: str,
        exit_reason: str,
        position_metrics: Optional[Dict[str, Any]],
        market_changes: Optional[Dict[str, Any]],
        entry_analysis: Dict[str, Any],
        exit_analysis: Dict[str, Any]
    ) -> List[str]:
        """
        提取可复用的教训

        Returns:
            教训列表
        """
        lessons = []

        # 1. 从入场分析提取教训
        if entry_analysis['is_correct']:
            if entry_analysis['score'] >= 60:
                lessons.append(f"✓ 入场信号'{entry_reason[:20]}...'可靠，可继续使用")
        else:
            lessons.append(f"✗ 入场信号'{entry_reason[:20]}...'不可靠，需谨慎")

        # 2. 从出场分析提取教训
        if position_metrics:
            max_profit_pct = position_metrics.get('max_profit_pct', 0)
            if max_profit_pct > 3 and profit_pct < max_profit_pct * 0.5:
                lessons.append(f"⚠️  下次在盈利 {max_profit_pct * 0.7:.1f}% 时考虑分批止盈")

            if profit_pct < -3:
                lessons.append(f"⚠️  设置更严格的止损点，避免亏损扩大")

        # 3. 从市场变化提取教训
        if market_changes:
            # TODO: 分析市场变化与交易结果的关系
            pass

        # 4. 通用教训
        if profit_pct > 5:
            lessons.append("✓ 成功案例，记住这种市场条件和操作方式")
        elif profit_pct < -5:
            lessons.append("✗ 避免在类似条件下重复此操作")

        return lessons

    def _generate_summary(
        self,
        profit_pct: float,
        entry_analysis: Dict[str, Any],
        exit_analysis: Dict[str, Any],
        lessons: List[str]
    ) -> str:
        """
        生成反思总结

        Returns:
            总结文本
        """
        result = '盈利' if profit_pct > 0 else '亏损'
        entry_score = entry_analysis['score']
        exit_score = exit_analysis['score']

        summary_parts = [
            f"结果: {result} {profit_pct:+.2f}%",
            f"入场评分: {entry_score}/100",
            f"出场评分: {exit_score}/100"
        ]

        # 关键问题
        if entry_analysis['weaknesses']:
            summary_parts.append(f"入场问题: {entry_analysis['weaknesses'][0]}")

        if exit_analysis['weaknesses']:
            summary_parts.append(f"出场问题: {exit_analysis['weaknesses'][0]}")

        # 改进建议
        if entry_score < 50:
            summary_parts.append("建议: 改进入场时机判断")
        elif exit_score < 50:
            summary_parts.append("建议: 改进出场时机和止盈止损策略")
        else:
            summary_parts.append("建议: 保持当前策略")

        return " | ".join(summary_parts)

    def format_reflection_for_log(self, reflection: Dict[str, Any]) -> str:
        """
        格式化反思报告为日志文本

        Args:
            reflection: 反思报告字典

        Returns:
            格式化的文本
        """
        lines = [
            f"【交易反思 - Trade #{reflection['trade_id']}】",
            f"结果: {reflection['result_type']} {reflection['profit_pct']:+.2f}%",
            "",
            f"开仓时判断:",
            f"  {reflection['entry_judgment'][:100]}...",
            "",
            f"实际走势:",
            f"  {reflection['actual_outcome'][:100]}...",
            "",
            f"入场评分: {reflection['entry_quality']['score']}/100"
        ]

        if reflection['entry_quality']['strengths']:
            lines.append(f"  ✓ {', '.join(reflection['entry_quality']['strengths'])}")

        if reflection['entry_quality']['weaknesses']:
            lines.append(f"  ✗ {', '.join(reflection['entry_quality']['weaknesses'])}")

        lines.append(f"\n出场评分: {reflection['exit_quality']['score']}/100")

        if reflection['exit_quality']['strengths']:
            lines.append(f"  ✓ {', '.join(reflection['exit_quality']['strengths'])}")

        if reflection['exit_quality']['weaknesses']:
            lines.append(f"  ✗ {', '.join(reflection['exit_quality']['weaknesses'])}")

        if reflection['lessons']:
            lines.append("\n教训:")
            for lesson in reflection['lessons']:
                lines.append(f"  {lesson}")

        lines.append(f"\n总结: {reflection['summary']}")

        return '\n'.join(lines)
