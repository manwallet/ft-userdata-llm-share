"""
交易复盘模块 - 自动交易复盘和经验提取
负责从完成的交易中提取结构化的经验教训
"""
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class TradeReviewer:
    """交易复盘器 - 自动复盘和教训提取"""

    def __init__(self):
        """初始化交易复盘器"""
        self.review_templates = self._init_review_templates()
        logger.info("交易复盘器已初始化")

    def _init_review_templates(self) -> Dict[str, str]:
        """初始化复盘模板"""
        return {
            "profit_drawdown": "盈利回撤{drawdown:.1f}% (从峰值{mfe:+.1f}%到最终{final:+.1f}%), {analysis}",
            "hold_too_long": "持仓{duration}但无显著进展, hold了{hold_count}次, 机会成本过高",
            "good_exit_timing": "在{profit:+.1f}%时主动止盈, 避免了后续{potential_loss}的风险",
            "poor_entry": "入场位置不佳, 开仓后立即出现{mae:+.1f}%浮亏",
            "good_patience": "经历{mae:+.1f}%浮亏后耐心持有, 最终实现{profit:+.1f}%盈利",
            "premature_exit": "过早退出, 离场后价格继续向预期方向移动",
            "trend_reversal_ignored": "趋势反转信号出现但未及时响应, 导致盈利回撤/亏损扩大"
        }

    def generate_trade_review(
        self,
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
        market_changes: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        生成交易复盘报告

        Args:
            pair: 交易对
            side: 方向 (long/short)
            entry_price: 入场价格
            exit_price: 出场价格
            entry_reason: 入场理由
            exit_reason: 出场理由
            profit_pct: 账户盈亏百分比（考虑杠杆）
            duration_minutes: 持仓时长
            leverage: 杠杆
            position_metrics: PositionTracker的指标数据
            market_changes: MarketStateComparator的变化数据

        Returns:
            复盘报告字典
        """
        try:
            # 基础信息
            result_type = "盈利" if profit_pct > 0 else "亏损"
            duration_str = self._format_duration(duration_minutes)

            # 提取position_metrics
            mfe = position_metrics.get("max_profit_pct", profit_pct) if position_metrics else profit_pct
            mae = position_metrics.get("max_loss_pct", profit_pct) if position_metrics else profit_pct
            hold_count = position_metrics.get("hold_count", 0) if position_metrics else 0
            drawdown_from_peak = position_metrics.get("drawdown_from_peak_pct", 0) if position_metrics else 0

            # 自动分析和教训提取
            lessons = []
            insights = []
            warnings = []

            # 1. 盈利回撤分析（核心问题）
            if mfe > 5 and (mfe - profit_pct) > 3:
                drawdown = mfe - profit_pct
                severity = "严重" if drawdown > 10 else "显著" if drawdown > 5 else "轻微"

                lesson = self.review_templates["profit_drawdown"].format(
                    drawdown=drawdown,
                    mfe=mfe,
                    final=profit_pct,
                    analysis=f"{severity}回撤"
                )
                lessons.append(lesson)

                if drawdown > 8:
                    warnings.append(
                        f"[关键教训] 盈利从{mfe:+.1f}%回撤到{profit_pct:+.1f}%, "
                        f"应在盈利峰值附近考虑部分止盈"
                    )

                # 分析hold次数与回撤的关系
                if hold_count >= 5:
                    insights.append(
                        f"hold了{hold_count}次期间盈利回撤{drawdown:.1f}%, "
                        f"可能存在确认偏差（寻找支持持仓的理由而忽略反转信号）"
                    )

            # 2. 持仓效率分析
            if duration_minutes > 60 * 24:  # 超过1天
                price_move_pct = abs(exit_price - entry_price) / entry_price * 100

                # 长时间持仓但收益不佳
                if abs(profit_pct) < 5 and duration_minutes > 60 * 48:
                    lesson = self.review_templates["hold_too_long"].format(
                        duration=duration_str,
                        hold_count=hold_count
                    )
                    lessons.append(lesson)
                    insights.append(
                        f"持仓{duration_str}但账户盈利仅{profit_pct:+.1f}%, "
                        f"价格波动{price_move_pct:.1f}%, 资金利用效率低"
                    )

            # 3. 入场质量分析
            if mae < -5:  # 开仓后立即大幅浮亏
                if profit_pct < 0:  # 且最终亏损
                    lesson = self.review_templates["poor_entry"].format(mae=mae)
                    lessons.append(lesson)
                    warnings.append(
                        f"入场后立即出现{mae:+.1f}%浮亏, 说明入场位置选择不佳或逆势交易"
                    )
                elif profit_pct > 0:  # 但最终盈利
                    lesson = self.review_templates["good_patience"].format(
                        mae=mae,
                        profit=profit_pct
                    )
                    lessons.append(lesson)
                    insights.append(
                        f"[正面案例] 虽入场位置不佳(浮亏{mae:+.1f}%), "
                        f"但耐心等待趋势展开最终盈利{profit_pct:+.1f}%"
                    )

            # 4. 退出质量分析
            exit_lower = exit_reason.lower()

            # 主动止盈退出
            if any(kw in exit_lower for kw in ["止盈", "盈利", "目标"]):
                if profit_pct > 5:
                    insights.append(
                        f"[正面案例] 主动在{profit_pct:+.1f}%止盈退出, 保住利润"
                    )

                    # 如果MFE显著高于最终profit，说明可能退出太早或再次入场
                    if mfe > profit_pct + 3:
                        insights.append(
                            f"注意: 最高曾到{mfe:+.1f}%, 退出时机可能偏早, "
                            f"但主动止盈总比盈利回撤好"
                        )

            # 被动止损/反转退出
            elif any(kw in exit_lower for kw in ["止损", "反转", "破位"]):
                if mfe > profit_pct + 5:  # 曾经有显著浮盈但被动退出
                    lesson = self.review_templates["trend_reversal_ignored"].format()
                    lessons.append(lesson)
                    warnings.append(
                        f"[关键教训] 曾有{mfe:+.1f}%盈利但未主动退出, "
                        f"等到趋势反转被动离场时只剩{profit_pct:+.1f}%"
                    )

            # 5. 市场环境变化分析（如果有数据）
            if market_changes:
                significant_changes = market_changes.get("significant_changes", [])
                if significant_changes:
                    insights.append("市场环境重大变化:")
                    for change in significant_changes[:3]:
                        insights.append(f"  - {change}")

            # 6. 综合评分
            score = self._calculate_trade_score(
                profit_pct=profit_pct,
                mfe=mfe,
                mae=mae,
                hold_count=hold_count,
                duration_minutes=duration_minutes
            )

            # 7. 生成简洁的经验教训总结
            concise_lesson = self._generate_concise_lesson(
                result_type=result_type,
                profit_pct=profit_pct,
                mfe=mfe,
                mae=mae,
                hold_count=hold_count,
                lessons=lessons,
                warnings=warnings
            )

            return {
                "pair": pair,
                "side": side,
                "result": result_type,
                "profit_pct": profit_pct,
                "duration": duration_str,
                "score": score,
                "mfe": mfe,
                "mae": mae,
                "hold_count": hold_count,
                "lessons": lessons,
                "insights": insights,
                "warnings": warnings,
                "concise_lesson": concise_lesson,
                "entry_reason": entry_reason,
                "exit_reason": exit_reason,
                "market_changes": significant_changes if market_changes else []
            }

        except Exception as e:
            logger.error(f"生成交易复盘失败: {e}")
            return {
                "error": str(e),
                "concise_lesson": f"{result_type} {profit_pct:+.1f}% - 复盘生成失败"
            }

    def _calculate_trade_score(
        self,
        profit_pct: float,
        mfe: float,
        mae: float,
        hold_count: int,
        duration_minutes: int
    ) -> float:
        """
        计算交易质量评分 (0-100)

        考虑因素:
        - 盈亏结果
        - 盈利回撤程度
        - 入场质量 (MAE)
        - 持仓效率 (hold次数, 时长)
        """
        score = 50.0  # 基准分

        # 1. 盈亏结果 (±30分)
        if profit_pct > 10:
            score += 30
        elif profit_pct > 5:
            score += 20
        elif profit_pct > 0:
            score += 10
        elif profit_pct > -5:
            score -= 10
        elif profit_pct > -10:
            score -= 20
        else:
            score -= 30

        # 2. 盈利回撤惩罚 (-20分)
        if mfe > 5:
            drawdown_pct = (mfe - profit_pct) / mfe * 100
            if drawdown_pct > 80:  # 回撤超过80%
                score -= 20
            elif drawdown_pct > 50:
                score -= 15
            elif drawdown_pct > 30:
                score -= 10

        # 3. 入场质量 (±10分)
        if mae > -2:  # 几乎没有浮亏
            score += 10
        elif mae > -5:
            score += 5
        elif mae < -10:
            score -= 5
        elif mae < -15:
            score -= 10

        # 4. 持仓效率 (-10分)
        if hold_count > 10:
            score -= 10
        elif hold_count > 5:
            score -= 5

        # 5. 时间效率
        hours = duration_minutes / 60
        if hours > 72 and abs(profit_pct) < 5:  # 超过3天但收益不佳
            score -= 5

        return max(0, min(100, score))

    def _generate_concise_lesson(
        self,
        result_type: str,
        profit_pct: float,
        mfe: float,
        mae: float,
        hold_count: int,
        lessons: List[str],
        warnings: List[str]
    ) -> str:
        """
        生成简洁的经验教训总结
        """
        parts = []

        # 核心结果
        parts.append(f"{result_type} {profit_pct:+.1f}%")

        # 最关键的教训
        if warnings:
            parts.append(warnings[0])
        elif lessons:
            parts.append(lessons[0])

        # 简要统计
        if mfe != profit_pct and abs(mfe - profit_pct) > 3:
            parts.append(f"(峰值{mfe:+.1f}%)")

        if hold_count >= 5:
            parts.append(f"(hold {hold_count}次)")

        return " | ".join(parts)

    def _format_duration(self, minutes: int) -> str:
        """格式化持仓时长"""
        if minutes < 60:
            return f"{minutes}分钟"
        elif minutes < 60 * 24:
            hours = minutes / 60
            return f"{hours:.1f}小时"
        else:
            days = minutes / (60 * 24)
            return f"{days:.1f}天"

    def format_review_report(self, review: Dict[str, Any]) -> str:
        """
        格式化复盘报告为可读文本

        Args:
            review: generate_trade_review返回的报告字典

        Returns:
            格式化的文本报告
        """
        if "error" in review:
            return f"复盘报告生成失败: {review['error']}"

        lines = [
            "=" * 60,
            f"【交易复盘报告】{review['pair']} - {review['side']}",
            "=" * 60,
            "",
            "基本信息:",
            f"  结果: {review['result']} {review['profit_pct']:+.2f}%",
            f"  持仓时长: {review['duration']}",
            f"  质量评分: {review['score']:.0f}/100",
            "",
            "盈亏演变:",
            f"  最大浮盈(MFE): {review['mfe']:+.2f}%",
            f"  最大浮亏(MAE): {review['mae']:+.2f}%",
            f"  最终结果: {review['profit_pct']:+.2f}%",
            f"  hold决策次数: {review['hold_count']}",
            ""
        ]

        # 警告（最重要）
        if review["warnings"]:
            lines.append("关键警告:")
            for warning in review["warnings"]:
                lines.append(f"  {warning}")
            lines.append("")

        # 经验教训
        if review["lessons"]:
            lines.append("经验教训:")
            for lesson in review["lessons"]:
                lines.append(f"  - {lesson}")
            lines.append("")

        # 深度洞察
        if review["insights"]:
            lines.append("深度洞察:")
            for insight in review["insights"]:
                lines.append(f"  {insight}")
            lines.append("")

        # 市场变化
        if review.get("market_changes"):
            lines.append("市场环境变化:")
            for change in review["market_changes"]:
                lines.append(f"  - {change}")
            lines.append("")

        # 入场/离场理由
        lines.extend([
            "决策记录:",
            f"  入场理由: {review.get('entry_reason', '')[:100]}...",
            f"  离场理由: {review.get('exit_reason', '')[:100]}...",
            "",
            "=" * 60
        ])

        return "\n".join(lines)
