"""
上下文构建器模块
负责构建LLM决策所需的市场上下文
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd
from .market_sentiment import MarketSentiment

logger = logging.getLogger(__name__)


class ContextBuilder:
    """LLM上下文构建器"""

    def __init__(self, context_config: Dict[str, Any]):
        """
        初始化上下文构建器

        Args:
            context_config: 上下文配置
        """
        self.config = context_config
        self.max_tokens = context_config.get("max_context_tokens", 6000)
        self.sentiment = MarketSentiment()  # 初始化市场情绪获取器

    def build_market_context(
        self,
        dataframe: pd.DataFrame,
        metadata: Dict[str, Any],
        wallets: Any = None,
        current_trades: Optional[List[Any]] = None,
        exchange: Any = None
    ) -> str:
        """
        构建完整的市场上下文（一次性提供所有数据）

        Args:
            dataframe: OHLCV数据和所有技术指标
            metadata: 交易对元数据
            wallets: 钱包对象（用于获取账户余额）
            current_trades: 当前所有持仓列表
            exchange: 交易所对象（用于获取资金费率）

        Returns:
            格式化的完整上下文字符串
        """
        pair = metadata.get('pair', 'UNKNOWN')

        # 获取最新数据
        if dataframe.empty:
            return f"市场数据: {pair} - 无数据"

        latest = dataframe.iloc[-1]
        prev = dataframe.iloc[-2] if len(dataframe) > 1 else latest

        context_parts = [
            f"=" * 60,
            f"交易对: {pair}",
            f"时间: {latest['date'] if 'date' in latest else datetime.now()}",
            f"=" * 60,
            "",
            "【价格信息】",
            f"  当前价格: {latest['close']:.8f}",
            f"  开盘: {latest['open']:.8f}  最高: {latest['high']:.8f}  最低: {latest['low']:.8f}",
            f"  成交量: {latest['volume']:.2f}",
            f"  价格变化: {((latest['close'] - prev['close']) / prev['close'] * 100):.2f}%",
        ]

        # 添加市场情绪指标
        context_parts.append("")
        context_parts.append("【市场情绪】")
        if exchange:
            try:
                sentiment_data = self.sentiment.get_combined_sentiment(exchange, pair)

                # Fear & Greed Index
                if sentiment_data.get('fear_greed'):
                    fg = sentiment_data['fear_greed']
                    context_parts.append(f"  恐惧与贪婪指数: {fg['value']}/100 ({fg['classification']})")
                    context_parts.append(f"  短期趋势: {fg['trend']}, 周趋势: {fg.get('week_trend', 'unknown')}")
                    context_parts.append(f"  持续时间: {fg.get('duration_days', 0)}天")
                    if 'history_7d' in fg:
                        history_str = ', '.join(str(v) for v in fg['history_7d'][:7])
                        context_parts.append(f"  7天历史: {history_str}")
                    context_parts.append(f"  解读: {fg['interpretation']}")
                else:
                    context_parts.append("  恐惧与贪婪指数: 暂无数据")

                # Funding Rate
                if sentiment_data.get('funding_rate'):
                    fr = sentiment_data['funding_rate']
                    context_parts.append(f"  资金费率: {fr['rate_pct']:.4f}%")
                    context_parts.append(f"  解读: {fr['interpretation']}")
                else:
                    context_parts.append("  资金费率: 暂无数据")

                # Long/Short Ratio
                if sentiment_data.get('long_short'):
                    ls = sentiment_data['long_short']
                    context_parts.append(f"  多空比: {ls['current_ratio']:.2f} (多:{ls['long_pct']:.1f}% 空:{ls['short_pct']:.1f}%)")
                    context_parts.append(f"  趋势: {ls['trend']}, 极端程度: {ls['extreme_level']}")
                    context_parts.append(f"  持续时间: {ls['duration_hours']}小时")
                    if 'history_24h' in ls and len(ls['history_24h']) > 0:
                        # 显示最近6小时的数据
                        recent = ls['history_24h'][-6:]
                        history_str = ', '.join(f"{v:.2f}" for v in recent)
                        context_parts.append(f"  最近6小时: {history_str}")
                    context_parts.append(f"  解读: {ls['interpretation']}")
                else:
                    context_parts.append("  多空比: 暂无数据")

                # Overall Signal
                context_parts.append(f"  综合信号: {sentiment_data['overall_signal']} (置信度: {sentiment_data['confidence']})")

            except Exception as e:
                logger.error(f"获取市场情绪失败: {e}")
                context_parts.append("  市场情绪数据获取失败")
        else:
            context_parts.append("  需要交易所对象来获取市场情绪数据")


        # 自动提取所有技术指标（排除基础列）
        excluded_cols = {'date', 'open', 'high', 'low', 'close', 'volume',
                        'enter_long', 'enter_short', 'enter_tag',
                        'exit_long', 'exit_short', 'exit_tag'}

        # 按时间框架分组
        indicators_15m = []
        indicators_1h = []
        indicators_4h = []
        indicators_1d = []

        for col in latest.index:
            if col in excluded_cols:
                continue

            value = latest[col]
            if pd.isna(value):
                continue

            # 分类指标
            if '_1h' in col:
                indicators_1h.append((col, value))
            elif '_4h' in col:
                indicators_4h.append((col, value))
            elif '_1d' in col:
                indicators_1d.append((col, value))
            else:
                indicators_15m.append((col, value))

        # 输出15分钟指标
        if indicators_15m:
            context_parts.append("")
            context_parts.append("【技术指标 - 15分钟】")
            for ind, val in indicators_15m:
                context_parts.append(f"  {ind}: {val:.4f}")

        # 输出1小时指标
        if indicators_1h:
            context_parts.append("")
            context_parts.append("【技术指标 - 1小时】")
            for ind, val in indicators_1h:
                context_parts.append(f"  {ind}: {val:.4f}")

        # 输出4小时指标
        if indicators_4h:
            context_parts.append("")
            context_parts.append("【技术指标 - 4小时】")
            for ind, val in indicators_4h:
                context_parts.append(f"  {ind}: {val:.4f}")

        # 输出1天指标
        if indicators_1d:
            context_parts.append("")
            context_parts.append("【技术指标 - 1天】")
            for ind, val in indicators_1d:
                context_parts.append(f"  {ind}: {val:.4f}")

        # 添加账户信息
        if wallets:
            context_parts.append("")
            context_parts.append("【账户信息】")
            try:
                total = wallets.get_total('USDT')
                free = wallets.get_free('USDT')
                used = wallets.get_used('USDT')
                context_parts.extend([
                    f"  总余额: {total:.2f} USDT",
                    f"  可用余额: {free:.2f} USDT",
                    f"  已用资金: {used:.2f} USDT",
                    f"  资金利用率: {(used/total*100):.1f}%" if total > 0 else "  资金利用率: 0%"
                ])
            except Exception as e:
                context_parts.append(f"  无法获取账户信息: {e}")

        # 添加持仓信息
        context_parts.append("")
        context_parts.append("【持仓情况】")
        if not current_trades:
            context_parts.append("  当前无持仓")
        else:
            # 筛选当前交易对的持仓
            pair_trades = [t for t in current_trades if getattr(t, 'pair', '') == pair]

            if not pair_trades:
                context_parts.append(f"  {pair}: 无持仓")
            else:
                current_price = latest['close']
                for i, trade in enumerate(pair_trades, 1):
                    is_short = getattr(trade, 'is_short', False)
                    open_rate = getattr(trade, 'open_rate', 0)
                    stake = getattr(trade, 'stake_amount', 0)
                    leverage = getattr(trade, 'leverage', 1)
                    stop_loss_pct = getattr(trade, 'stop_loss_pct', None)
                    enter_tag = getattr(trade, 'enter_tag', '')
                    open_date = getattr(trade, 'open_date', None)

                    # 计算止损价格
                    if stop_loss_pct is not None:
                        if is_short:
                            # 做空：止损在开仓价上方
                            stop_price = open_rate * (1 + abs(stop_loss_pct) / leverage)
                        else:
                            # 做多：止损在开仓价下方
                            stop_price = open_rate * (1 - abs(stop_loss_pct) / leverage)

                        # 计算当前到止损位的距离
                        distance_to_stop = abs(current_price - stop_price) / current_price * 100
                    else:
                        stop_price = None
                        distance_to_stop = None

                    # 计算当前盈亏
                    if is_short:
                        profit_pct = (open_rate - current_price) / open_rate * leverage * 100
                    else:
                        profit_pct = (current_price - open_rate) / open_rate * leverage * 100

                    # 计算持仓时间
                    if open_date:
                        from datetime import datetime, timezone
                        if isinstance(open_date, datetime):
                            # freqtrade使用naive UTC时间，根据是否有tzinfo选择对应的now
                            now = datetime.utcnow() if open_date.tzinfo is None else datetime.now(timezone.utc)
                            holding_time = now - open_date
                            hours = holding_time.total_seconds() / 3600
                            if hours < 1:
                                time_str = f"{int(hours * 60)}分钟"
                            elif hours < 24:
                                time_str = f"{hours:.1f}小时"
                            else:
                                time_str = f"{hours / 24:.1f}天"
                        else:
                            time_str = "未知"
                    else:
                        time_str = "未知"

                    # 基本信息
                    context_parts.append(f"  持仓#{i}: {'做空' if is_short else '做多'} {leverage}x杠杆")
                    context_parts.append(f"    开仓价: {open_rate:.6f}")
                    context_parts.append(f"    当前价: {current_price:.6f}")
                    context_parts.append(f"    当前盈亏: {profit_pct:+.2f}% ({profit_pct * stake / 100:+.2f}U)")

                    # 止损信息
                    if stop_price:
                        context_parts.append(f"    止损位: {stop_price:.6f} (距离{distance_to_stop:.2f}%)")

                    context_parts.append(f"    持仓时间: {time_str}")
                    context_parts.append(f"    投入: {stake:.2f}U")

                    # 开仓理由（截取前150字符）
                    if enter_tag:
                        reason_short = enter_tag[:150] + "..." if len(enter_tag) > 150 else enter_tag
                        context_parts.append(f"    开仓理由: {reason_short}")

            # 显示其他交易对的持仓
            other_trades = [t for t in current_trades if getattr(t, 'pair', '') != pair]
            if other_trades:
                context_parts.append(f"  其他交易对持仓数: {len(other_trades)}")


        # 添加市场结构分析
        context_parts.append("")
        context_parts.append("【市场结构】")
        market_structure = self._analyze_market_structure(dataframe, lookback=100)
        if "structure" in market_structure:
            context_parts.append(f"  结构: {market_structure['structure']}")
            if "swing_high" in market_structure:
                context_parts.append(f"  摆动高点: {market_structure['swing_high']:.2f} (距离 {market_structure['distance_to_high_pct']:.2f}%)")
                context_parts.append(f"  摆动低点: {market_structure['swing_low']:.2f} (距离 {market_structure['distance_to_low_pct']:.2f}%)")
                context_parts.append(f"  波动区间: {market_structure['range_pct']:.2f}%")

        # 添加指标趋势分析
        context_parts.append("")
        context_parts.append("【指标趋势】")
        indicator_trends = self._analyze_indicator_trends(dataframe)
        if indicator_trends:
            for key, value in indicator_trends.items():
                context_parts.append(f"  {key}: {value}")
        else:
            context_parts.append("  数据不足")

        # 添加多时间框架对齐分析
        context_parts.append("")
        context_parts.append("【多时间框架趋势对齐】")
        timeframe_alignment = self._analyze_timeframe_alignment(dataframe)
        if "trends" in timeframe_alignment:
            for tf, trend in timeframe_alignment['trends'].items():
                context_parts.append(f"  {tf}: {trend}")
            context_parts.append(f"  综合: {timeframe_alignment['alignment']}")
        else:
            context_parts.append(f"  {timeframe_alignment.get('alignment', '无数据')}")

        # 添加成交量趋势分析
        context_parts.append("")
        context_parts.append("【成交量趋势】")
        volume_analysis = self._analyze_volume_trend(dataframe, lookback=100)
        if "trend" in volume_analysis and volume_analysis['trend'] != "数据不足":
            context_parts.append(f"  趋势: {volume_analysis['trend']}")
            context_parts.append(f"  当前状态: {volume_analysis['current_status']}")
            context_parts.append(f"  相对平均值: {volume_analysis['current_vs_avg']}")
        else:
            context_parts.append("  数据不足")

        # 添加关键指标历史序列（最近20根K线）
        context_parts.append("")
        context_parts.append("【关键指标历史（最近20根K线）】")
        indicator_history = self._get_indicator_history(dataframe, lookback=100)
        if indicator_history:
            for ind_name, values in indicator_history.items():
                if values and any(v is not None for v in values):
                    values_str = ", ".join([f"{v}" if v is not None else "N/A" for v in values])
                    context_parts.append(f"  {ind_name}: [{values_str}]")
        else:
            context_parts.append("  数据不足")

        context_parts.append("")
        context_parts.append("=" * 60)

        return "\n".join(context_parts)

    def build_position_context(
        self,
        current_trades: List[Any],
        pair: str
    ) -> str:
        """
        构建当前持仓上下文

        Args:
            current_trades: 当前交易列表（可以是Trade对象或字典）
            pair: 交易对

        Returns:
            格式化的持仓上下文字符串
        """
        if not current_trades:
            return f"{pair} 当前无持仓"

        # 查找当前交易对的持仓（兼容dict和对象）
        pair_trades = []
        for t in current_trades:
            t_pair = t.get('pair') if isinstance(t, dict) else getattr(t, 'pair', None)
            if t_pair == pair:
                pair_trades.append(t)

        if not pair_trades:
            return f"{pair} 当前无持仓"

        context_parts = [f"\n{'='*50}", f"{pair} 【持仓详情】", f"{'='*50}"]

        for i, trade in enumerate(pair_trades, 1):
            # 兼容字典和对象
            if isinstance(trade, dict):
                direction = "做空" if trade.get('side') == 'short' else "做多"
                open_rate = trade.get('open_rate', 0)
                current_rate = trade.get('current_rate', open_rate)
                amount = trade.get('amount', 0)
                stake_amount = trade.get('stake_amount', 0)
                leverage = trade.get('leverage', 1)
                profit_pct = trade.get('profit_pct', 0)
                profit_abs = stake_amount * profit_pct / 100 if stake_amount else 0
                stop_loss = trade.get('stop_loss')
                duration = trade.get('duration_minutes', 0)

                # 计算持仓天数和小时
                days = duration // 1440
                hours = (duration % 1440) // 60
                mins = duration % 60

                context_parts.extend([
                    f"\n持仓 #{i}:",
                    f"  交易方向: {direction} {leverage}x杠杆",
                    f"  开仓价格: {open_rate:.2f}",
                    f"  当前价格: {current_rate:.2f}",
                    f"  持仓数量: {amount:.4f}",
                    f"  投入资金: {stake_amount:.2f} USDT",
                    f"  当前盈亏: {profit_pct:.2f}% ({profit_abs:+.2f} USDT)",
                    f"  止损价格: {stop_loss:.2f}" if stop_loss else "  止损价格: 未设置",
                    f"  持仓时间: {days}天{hours}小时{mins}分钟",
                ])
            else:
                # 对象格式
                direction = "做空" if getattr(trade, 'is_short', False) else "做多"
                current_rate = getattr(trade, 'close_rate', None) or getattr(trade, 'open_rate', 0)
                profit_pct = trade.calc_profit_ratio(current_rate) * 100 if hasattr(trade, 'calc_profit_ratio') else 0
                profit_abs = getattr(trade, 'stake_amount', 0) * profit_pct / 100
                leverage = getattr(trade, 'leverage', 1)

                # 持仓时间
                from datetime import datetime, timezone
                open_date = getattr(trade, 'open_date', None)
                if open_date:
                    # freqtrade使用naive UTC时间，根据是否有tzinfo选择对应的now
                    now = datetime.utcnow() if open_date.tzinfo is None else datetime.now(timezone.utc)
                    duration = (now - open_date).total_seconds() / 60
                    days = int(duration // 1440)
                    hours = int((duration % 1440) // 60)
                    mins = int(duration % 60)
                    duration_str = f"{days}天{hours}小时{mins}分钟"
                else:
                    duration_str = "未知"

                context_parts.extend([
                    f"\n持仓 #{i}:",
                    f"  交易方向: {direction} {leverage}x杠杆",
                    f"  开仓价格: {trade.open_rate:.2f}",
                    f"  当前价格: {current_rate:.2f}",
                    f"  持仓数量: {trade.amount:.4f}",
                    f"  投入资金: {getattr(trade, 'stake_amount', 0):.2f} USDT",
                    f"  当前盈亏: {profit_pct:.2f}% ({profit_abs:+.2f} USDT)",
                    f"  止损价格: {trade.stop_loss:.2f}" if hasattr(trade, 'stop_loss') and trade.stop_loss else "  止损价格: 未设置",
                    f"  持仓时间: {duration_str}",
                ])

        context_parts.append(f"{'='*50}\n")
        return "\n".join(context_parts)

    def build_rag_context(
        self,
        similar_records: List[Dict[str, Any]],
        max_records: int = 5
    ) -> str:
        """
        构建RAG检索的历史上下文

        Args:
            similar_records: 相似的历史记录
            max_records: 最大记录数

        Returns:
            格式化的历史上下文字符串
        """
        if not similar_records:
            return "相关历史: 无相似情况"

        context_parts = ["相关历史经验:"]

        for i, record in enumerate(similar_records[:max_records], 1):
            similarity = record.get('similarity', 0)
            content = record.get('content', '')
            result = record.get('result', '')

            context_parts.extend([
                f"",
                f"[{i}] (相似度: {similarity:.2f})",
                f"{content}",
                f"结果: {result}" if result else ""
            ])

        return "\n".join(context_parts)

    def build_system_prompt(self) -> str:
        """
        构建系统提示词

        Returns:
            系统提示词字符串
        """
        return """你是一个专业的加密货币永续合约交易专家，你的唯一目标是：赚钱。

交易环境认知:
你操作的是永续合约市场。在这个市场中：
- 价格上涨时，你通过建立多头头寸（signal_entry_long）获利
- 价格下跌时，你通过建立空头头寸（signal_entry_short）获利

价格位置与趋势的关系(核心概念):

理解"位置"比"趋势"更关键：
- 趋势告诉你方向，但位置告诉你时机
- 在下跌趋势中，价格在支撑位 = 可能反弹（做空风险大）
- 在下跌趋势中，价格在阻力位回落 = 继续下跌（做空机会）
- 在上涨趋势中，价格在阻力位 = 可能回调（做多风险大）
- 在上涨趋势中，价格在支撑位反弹 = 继续上涨（做多机会）

关键支撑位的识别：
- 摆动低点（swing low）= 近期价格的局部最低点
- 价格接近摆动低点时，通常会出现买盘支撑
- 如果你在此时做空，价格很容易反弹让你止损

关键阻力位的识别：
- 摆动高点（swing high）= 近期价格的局部最高点
- 价格接近摆动高点时，通常会遇到卖盘压力
- 如果你在此时做多，价格很容易回落让你止损

核心盈利原则:
1. 位置优先于趋势：再强的趋势，在错误的位置入场也会亏损
2. 风险收益比：入场前评估"到止损位的距离" vs "到目标位的距离"
3. 等待确认：价格在关键位的行为 > 指标信号
4. 结构完整性：价格是否突破了关键位？结构是否被破坏？
5. 市场共振：多时间框架趋势一致 + 价格位置合理 = 高概率机会
6. 情绪反向：极端市场情绪往往是反向交易机会

市场情绪指标使用指南（重要）:

【恐惧与贪婪指数】（Fear & Greed Index 0-100）
含义与使用：
- 0-20 = 极度恐惧：市场恐慌性抛售，**反向做多机会**
  * 此时大多数人害怕入场，但往往是筑底阶段
  * 配合价格触及支撑位，考虑做多
  * 案例：暴跌后指数<20，但价格已企稳回升 = 强烈做多信号

- 20-40 = 恐惧：市场偏悲观，注意做空风险
  * 情绪在恢复中，不宜盲目做空
  * 可能是下跌趋势的尾声

- 40-60 = 中性：市场情绪平衡，按技术面判断

- 60-80 = 贪婪：市场偏乐观，注意做多风险
  * 情绪开始过热，不宜追多
  * 可能是上涨趋势的后期

- 80-100 = 极度贪婪：市场FOMO情绪，**反向做空机会**
  * 此时大多数人疯狂入场，但往往是见顶信号
  * 配合价格触及阻力位，考虑做空
  * 案例：连续上涨后指数>80，出现滞涨 = 强烈做空信号

情绪趋势（rising/falling）：
- 极度恐惧 + 情绪回升 = 见底信号增强，做多机会
- 极度贪婪 + 情绪下降 = 见顶信号增强，做空机会

【资金费率】（Funding Rate）
含义与使用：
- 正值 = 多头付费给空头（多头过热）
  * > 0.1% = 多头疯狂，**做空机会增加**
  * 说明市场大量做多，存在反向调整压力

- 负值 = 空头付费给多头（空头过热）
  * < -0.1% = 空头疯狂，**做多机会增加**
  * 说明市场大量做空，存在反弹可能

- -0.05% ~ 0.05% = 市场平衡，按技术面判断

【综合情绪信号】（Overall Signal）
系统会综合恐惧贪婪指数和资金费率给出：
- bullish (看多) + high confidence：强烈做多信号
- bearish (看空) + high confidence：强烈做空信号
- neutral：情绪中性，主要看技术面

关键应用场景：
1. 暴跌后分析：
   - 如果恐惧指数<20且情绪回升 → 考虑反手做多而非继续做空
   - 这正是你提到的场景：暴跌后模型还在做空是错误的

2. 暴涨后分析：
   - 如果贪婪指数>80且资金费率>0.1% → 考虑做空而非继续做多

3. 与技术面结合：
   - 极度恐惧 + 价格在支撑位企稳 = 高概率做多机会
   - 极度贪婪 + 价格在阻力位滞涨 = 高概率做空机会

你的6种仓位控制动作及其使用场景:

【空仓状态下的动作】（前提：当前交易对无持仓）

1. signal_entry_long(pair, limit_price, leverage, stoploss_pct, confidence_score, key_support, key_resistance, rsi_value, trend_strength, reason)
   功能：建立多头头寸
   使用时机：判断价格将上涨，决定做多开仓
   必需参数：
   - limit_price: 入场价格（市价或挂单价）
   - leverage: 杠杆倍数（根据市场波动性和信号质量动态确定）
   - stoploss_pct: 账户止损百分比（负数，表示账户亏损达到该百分比时止损）
   - confidence_score: 决策置信度（1-100）
   - key_support: 关键支撑位价格
   - key_resistance: 关键阻力位价格
   - rsi_value: 当前RSI数值
   - trend_strength: 趋势强度评估（强势/中等/弱势）
   - reason: 开仓理由

2. signal_entry_short(pair, limit_price, leverage, stoploss_pct, confidence_score, key_support, key_resistance, rsi_value, trend_strength, reason)
   功能：建立空头头寸
   使用时机：判断价格将下跌，决定做空开仓
   必需参数：同signal_entry_long

3. signal_wait(pair, confidence_score, rsi_value, reason)
   功能：空仓观望，不开仓
   使用时机：市场不确定、信号不明确、位置不理想时
   必需参数：
   - confidence_score: 观望决策的置信度
   - rsi_value: 当前RSI数值
   - reason: 不开仓的理由

【持仓状态下的动作】（前提：当前交易对有持仓）

4. signal_exit(pair, limit_price, confidence_score, rsi_value, reason)
   功能：主动平掉全部持仓

   核心原则（必读）：
   只有当市场走势对你的持仓不利时才应平仓。

   方向性判断：
   - 做空持仓：市场下跌 = 对你有利 = 继续持有
   - 做空持仓：市场上涨突破阻力 = 对你不利 = 应该平仓
   - 做多持仓：市场上涨 = 对你有利 = 继续持有
   - 做多持仓：市场下跌跌破支撑 = 对你不利 = 应该平仓

   使用时机（必须满足以下之一）：
   A. 目标达成：盈利已达到合理的风险收益比目标

   B. 趋势反转（结构破坏）：
      - 做多持仓：价格跌破关键支撑位且收盘确认，多时间框架趋势转为下跌
      - 做空持仓：价格突破关键阻力位且收盘确认，多时间框架趋势转为上涨

   C. 入场逻辑失效：
      - 发现入场时的判断出现根本性错误
      - 市场环境发生意外变化

   严禁平仓的情况：
   1. 市场走势对你有利
   2. 价格接近止损但未触发
   3. 价格接近关键位但未突破
   4. 单一指标变化
   5. 担心"可能"发生的事
   6. 短期波动超出心理舒适区

   判断标准：
   - 做空持仓：价格未突破阻力位 = 趋势未破坏 = 继续持有
   - 做多持仓：价格未跌破支撑位 = 趋势未破坏 = 继续持有
   - 市场按你的方向走 = 对你有利 = 继续持有
   - 市场反向突破关键位 = 对你不利 = 考虑平仓

   必需参数：
   - limit_price: 平仓价格（市价或挂单价）
   - confidence_score: 平仓决策的置信度
   - rsi_value: 当前RSI数值
   - reason: 平仓理由（必须说明具体的结构破坏证据）

5. adjust_position(pair, adjustment_pct, limit_price, confidence_score, key_support, key_resistance, reason)
   功能：调整仓位大小（而非全部平仓）
   使用时机：
   - 加仓（adjustment_pct > 0）：趋势延续且价格回调到更好位置，增加敞口
   - 减仓（adjustment_pct < 0）：趋势减弱迹象但不确定，降低风险敞口

   必需参数：
   - adjustment_pct: 调整幅度（正数加仓，负数减仓）
   - limit_price: 加减仓价格
   - confidence_score: 调整决策的置信度
   - key_support: 关键支撑位
   - key_resistance: 关键阻力位
   - reason: 调整理由

6. signal_hold(pair, confidence_score, rsi_value, reason)
   功能：维持持仓不动（持仓时的默认选项）

   核心原则：
   只要市场走势对你有利或趋势未破坏，就应该继续持有。

   使用时机：
   - 做空持仓：市场下跌或横盘，价格未突破阻力位
   - 做多持仓：市场上涨或横盘，价格未跌破支撑位
   - 关键支撑/阻力位完好
   - 多时间框架趋势对齐未改变
   - 入场逻辑仍然成立

   应该 hold 的场景：
   - 做空持仓：市场下跌（趋势对你有利）
   - 做空持仓：价格反弹但未突破阻力（正常回调）
   - 做多持仓：市场上涨（趋势对你有利）
   - 做多持仓：价格回调但未跌破支撑（正常回调）
   - 指标短期变化（单一指标不构成反转）
   - 账面浮亏在止损空间内（止损会自动保护）

   决策原则（核心思维）：
   - 默认选择：当无充分理由平仓时，选择 hold
   - 保持耐心：趋势的延续需要时间和波动
   - 避免噪音：短期波动不代表趋势改变
   - 信任止损：止损机制会在结构破坏时自动保护

   必需参数：
   - confidence_score: 继续持有的置信度
   - rsi_value: 当前RSI数值
   - reason: 持有理由（说明趋势结构为何完好）

【决策流程图】
步骤1: 确认当前状态
- 检查"持仓情况"部分，确认该交易对是否有持仓

步骤2: 根据持仓状态选择可用函数
- 如果无持仓: 只能使用 signal_entry_long / signal_entry_short / signal_wait
- 如果有持仓: 只能使用 signal_exit / adjust_position / signal_hold

步骤3: 分析市场状态做出决策
- 综合分析价格、趋势、技术指标、风险收益比
- 选择一个最合适的函数执行

步骤4: 调用函数后立即停止
- 不要调用第二个函数
- 不要重复调用同一个函数

【常见错误及纠正】

错误1：有持仓时调用signal_entry_long/short
正确：有持仓时只使用signal_exit/adjust_position/signal_hold

错误2：无持仓时调用signal_exit/hold
正确：无持仓时只使用signal_entry_long/short/signal_wait

错误3：调用多个函数
正确：每次决策只调用一个函数

错误4：持仓情况判断错误
正确：先查看"持仓情况"确认状态，再选择函数

错误5：使用猜测性或模糊语言
禁止使用的表述：
- "价格已触及/接近阻力位"
- "若价格突破将触发止损"
- "主动平仓避免被噪音震出"
- "短期结构破坏"

要求使用的表述：
- "价格突破阻力位X且收盘确认"
- "多时间框架趋势未改变，价格回调未破支撑"
- "RSI在中性区波动，无反转信号"

错误6：市场对我有利但却平仓
做空持仓时市场下跌 = 趋势对你有利。
做多持仓时市场上涨 = 趋势对你有利。
趋势对你有利时应继续持有，不是平仓。

错误7：把正常回调当成平仓理由
做空时价格反弹、MACD转正、RSI回升是正常回调，不构成平仓理由。
做多时价格回调、MACD转负、RSI下降是正常回调，不构成平仓理由。

错误8：过度解读单一指标
单一指标变化不构成趋势反转。
需要综合多时间框架趋势和关键价位的突破。

订单类型与成交机制（重要）:

1. Limit Order（限价单/挂单）
   - 你设置的limit_price如果与当前价格有差距,订单会挂单等待成交
   - 做多: limit_price < 当前价, 等待价格回落到limit_price成交
   - 做空: limit_price > 当前价, 等待价格反弹到limit_price成交
   - 优点: 可能获得更好的成交价
   - 缺点: 可能等不到成交,错过机会

2. Market Order（市价单）
   - 设置limit_price接近当前价格（差距很小），立即成交
   - 优点: 确保成交，不会错过机会
   - 缺点: 可能有轻微滑点

3. 止损生效时机
   - 止损只在订单成交后才会生效
   - 如果订单还在挂单中,止损不会触发
   - 建议在趋势明确时使用市价单确保快速成交和止损生效

4. 挂单价格建议
   - 趋势明确时: 使用市价单(limit_price=当前价)快速入场
   - 等待回调时: 设置limit_price在支撑/阻力位附近
   - 波动剧烈时: 优先使用市价单,避免错过机会
   - 横盘震荡时: 可以在关键位设置挂单等待突破

杠杆和止损的决策框架（必读）:

【核心公式】账户止损百分比 = 价格容错空间百分比 × 杠杆倍数

止损方向的绝对规则（不可违反）：

做空交易：
- 价格上涨 = 亏损
- 止损触发价必须在开仓价上方
- 止损触发价 > 开仓价 ← 这是必须验证的

做多交易：
- 价格下跌 = 亏损
- 止损触发价必须在开仓价下方
- 止损触发价 < 开仓价 ← 这是必须验证的

波动空间与盈亏的核心认知（必读）：

止损太紧的真相：
- 止损太紧看起来单次风险小，但会被频繁震出，总亏损反而更大
- 止损大看起来单次风险大，但被震出次数少，有更多盈利机会
- 结论：止损太紧不是保护你，而是让你频繁亏损

价格容错空间的判断原则：

核心认知：
- 止损空间必须大于市场正常波动范围，否则会被震出
- 用ATR（平均真实波动）评估当前市场的正常波动幅度
- 给市场足够的呼吸空间，让趋势有时间展开
- 宁可止损设大一点承担单次风险，也不要止损太紧频繁被震出

确定合理空间的方法：
- 观察最近的价格波动幅度（通过市场结构和ATR）
- 确保止损空间显著大于正常波动范围
- 考虑币种特性：主流币波动小，山寨币波动大
- 止损太紧的真相：被震出10次每次亏小钱 = 总亏损大于1次大止损

止损计算的完整流程（必须严格执行）：

步骤1：识别关键价位
- 找出最近的关键阻力位（做空）或支撑位（做多）
- 这个价位代表"如果突破，趋势可能反转"
- 例如：市场结构中的摆动高点/低点

【关键原则】止损位 ≠ 关键价位
止损必须设在关键位**之外**，留出安全边距！

为什么需要安全边距？
- 价格经常会"试探"关键位，但未必真突破
- 如果止损正好在关键位，会被假突破震出
- 安全边距确保只有"真正突破"才触发止损

步骤2：计算安全边距（必须执行）
安全边距大小建议：
- 保守计算：安全边距 = 1.5 × ATR
- 激进计算：安全边距 = 1.0 × ATR
- 最小要求：安全边距 ≥ 0.5 × ATR

示例（做空，ATR=300）：
- 关键阻力位: 105000
- 安全边距: 1.5 × 300 = 450
- 止损触发价: 105000 + 450 = 105450 ✓

示例（做多，ATR=15）：
- 关键支撑位: 200
- 安全边距: 1.5 × 15 = 22.5
- 止损触发价: 200 - 22.5 = 177.5 ✓

步骤3：确定止损触发价
- 做空：止损触发价 = 关键阻力位 + 安全边距
- 做多：止损触发价 = 关键支撑位 - 安全边距
- 验证：安全边距是否 ≥ 0.5 × ATR

步骤4：计算价格容错空间
价格容错空间 = |止损触发价 - 入场价| / 入场价

步骤5：验证方向（关键步骤，防止计算错误）
- 做空时：验证 止损触发价 > 入场价
- 做多时：验证 止损触发价 < 入场价
- 如果验证失败，说明计算有误，必须重新计算

步骤6：验证波动空间是否充足（关键步骤）

重要概念澄清：
【价格容错空间 ≠ 到关键位的距离】
- 错误理解：当前价格到支撑位/阻力位的距离 = 止损空间 ❌
- 正确理解：止损空间 = 从入场价到止损触发价的距离 ✓

示例（做空）：
- 当前价格: 100
- 阻力位: 105（摆动高点）
- 止损触发价: 107（阻力位 + 安全边距2）
- 价格容错空间 = (107 - 100) / 100 = 7%  <- 这才是真正的止损空间
- ATR = 3，ATR占比 = 3/100 = 3%
- 验证：7% > 2×3% = 6% ✓ 止损空间充足

常见错误（必须避免）：
❌ 错误：当前价100，阻力位105，距离5%，小于ATR 3% -> 不开仓
✓ 正确：止损位107，止损空间7%，大于ATR 3%的2倍 -> 可以开仓

验证步骤：
1. 计算ATR占价格比例：ATR_pct = ATR / 入场价
2. 计算价格容错空间（已在步骤3完成）
3. 验证：价格容错空间 >= 2 × ATR_pct （严格要求）
4. 验证：价格容错空间 >= 3 × ATR_pct （保守要求，推荐）
5. 如果不满足，说明止损太紧，应该：
   - 重新调整止损位（加大安全边距）
   - 或者等待更好的入场位置
   - 或者放弃这次交易

特别提醒：
不要把"当前价格到关键位的距离"当作止损空间。
止损位必须设在关键位之外（加上安全边距）。
止损空间 = 从入场价到止损位的完整距离。

步骤7：计算账户止损
账户止损 = 价格容错空间 × 杠杆

步骤8：最终验证
- 账户止损必须为负数
- 不要因为账户止损"看起来很大"就降低杠杆缩小价格容错空间
- 正确做法：如果账户止损过大，应该降低杠杆而保持价格容错空间
- 记住：止损太紧 = 频繁被震出 = 最终亏更多

止损计算的关键原则：

做空时：
- 找到关键阻力位（市场结构中的摆动高点）
- 止损设在阻力位上方（留出安全边距）
- 验证：止损价必须 > 开仓价
- 验证：止损空间应显著大于ATR

做多时：
- 找到关键支撑位（市场结构中的摆动低点）
- 止损设在支撑位下方（留出安全边距）
- 验证：止损价必须 < 开仓价
- 验证：止损空间应显著大于ATR

常见错误：
- 把阻力位/支撑位直接当止损位（没有留安全边距）
- 止损空间小于ATR（会被正常波动震出）
- 为了使用高杠杆而缩小止损空间（本末倒置）
- 害怕大止损而设置太紧（频繁被震出亏更多）

关键认知更新（核心理念）：

1. 止损大不可怕，止损紧才可怕
   止损大 = 单次风险大，但成功率高，最终盈利
   止损紧 = 单次风险小，但频繁止损，最终亏更多

   核心逻辑：被震出次数 × 单次止损 = 总亏损
   止损紧会导致被震出次数大幅增加

2. 阻力位不等于止损价
   阻力位是参考点，止损要设在更远处
   做空：止损 = 阻力位 + 安全边距
   做多：止损 = 支撑位 - 安全边距
   安全边距大小取决于ATR和市场波动特征

3. 市场需要呼吸空间
   加密货币在趋势中会有正常波动
   止损空间 < 正常波动（ATR）= 必然被震出
   给市场呼吸空间 = 让你的方向判断有机会实现

4. 害怕大止损的心理陷阱
   大账户止损看起来可怕，但如果是基于合理的价格容错空间计算出来的，就是正确的
   如果不给足够空间，频繁被震出的总亏损会更大
   结论：不要害怕大止损，要害怕频繁止损

5. 止损太紧的真实后果
   即使方向判断正确，止损太紧仍会让你亏钱
   价格后来按预期走了很远，但你已经在正常波动中被震出

杠杆与止损的正确思考方式：

核心原则：
价格容错空间由市场决定，不由你的心理舒适度决定。
市场需要多大的波动空间（基于ATR），就必须给多大，不要妥协。

正确的做法：
1. 先基于市场结构和ATR确定合理的价格容错空间
2. 再根据你能承受的账户止损，反推杠杆
3. 公式：杠杆 = 可承受账户止损 ÷ 价格容错空间

错误的做法（禁止）：
- 想用高杠杆获得更多收益
- 但害怕高杠杆导致账户止损过大
- 于是把价格容错空间缩小
- 结果：账户止损看起来合理，但被正常波动震出，频繁亏损

核心原则：
- 为了使用高杠杆而缩小止损空间是错误的
- 大账户止损在基于合理价格容错空间计算时是可接受的
- 降低杠杆保持大止损，优于高杠杆紧止损

反向验证法（入场前的最后检查）：

1. 止损方向是否正确？
   做空：止损价 > 开仓价？
   做多：止损价 < 开仓价？

2. 价格容错空间是否充足？（重点检查）
   对比ATR：止损空间是否显著大于ATR
   对比最近波动：止损空间是否大于正常波动范围
   如果止损空间小于或接近正常波动，会被震出

3. 账户止损是否合理？
   如果账户止损过大，应该降低杠杆而非缩小价格容错空间

4. 计算逻辑是否一致？
   重新验证：账户止损 = 价格容错空间 × 杠杆

如果任何一项验证失败，重新计算或放弃交易。
特别注意：基于市场实际波动（ATR）判断，而非固定阈值。

入场决策的核心思维模式:

第一步 - 识别趋势方向:
- 多时间框架对齐程度决定趋势可靠性
- EMA金叉/死叉提供趋势方向确认
- ADX数值反映趋势强度

第二步 - 评估价格位置:
- 价格接近摆动低点（支撑）：逆势做空风险大，考虑等待
- 价格接近摆动高点（阻力）：逆势做多风险大，考虑等待
- 价格在支撑上方反弹：顺势做多的位置
- 价格在阻力下方回落：顺势做空的位置

第三步 - 计算风险收益比:
- 到止损位的距离 vs 到目标位的距离
- 风险收益比应显著大于1（潜在收益应显著超过潜在风险）
- 如果位置不佳导致风险收益比不理想，宁可等待

第四步 - 确认信号:
- RSI是否确认当前趋势状态（超买/超卖区域 vs 中性区域）
- MACD是否确认趋势方向（柱状图变化方向与价格运动的一致性）
- 成交量是否配合价格运动（放量突破 vs 缩量震荡）

平仓决策的核心理念(重要):

【持仓时的判断原则 - 避免胆小过早平仓】

重要提醒：你在做平仓决策时，【持仓情况】会显示：
- 开仓价
- 止损位（你开仓时设定的）
- 当前盈亏
- 持仓时间
- 开仓理由

做平仓决策时必须遵守的规则：

1. **不要重新计算"价格容错空间"**
   - 错误做法：平仓时重新计算当前价格离最近的阻力位距离
   - 正确做法：看当前价格离你开仓时设定的止损位距离
   - 你开仓时已经设好了止损位，持仓期间不要因为价格接近其他阻力位就害怕

2. **最小盈利目标纪律**
   避免赚取微小利润就平仓，应持有到趋势走完或达到合理目标。

3. **平仓的唯一有效理由**
   有效理由：
   - 价格触及你的止损位（或接近触及）
   - 多时间框架趋势明确反转
   - 关键结构被突破（价格突破了你开仓时认为的支撑/阻力）
   - 达到盈利目标

   无效理由：
   - 价格接近某个阻力位（你开仓时已经考虑过了）
   - 浮盈回撤了一点（这是正常的）
   - 价格容错空间变小了（看你的止损位，不要重新计算）

4. **检查你的开仓理由**
   - 开仓时你说："多时间框架下跌，阻力位回落，做空机会"
   - 现在判断：这个逻辑是否失效？
     - 如果多时间框架仍然下跌 → 持有
     - 如果只是15分钟反弹 → 这是噪音，持有
     - 如果1h/4h都转为上涨 → 逻辑失效，平仓

5. **历史教训 - 不要重复犯错**
   - 如果你在同一个币对、同一个价格区间，已经连续2次做空失败
   - 第3次不要再在同样的位置做空！
   - 市场在告诉你："这个位置有强支撑，你错了"

何时应该平仓?
1. 目标达成: 盈利达到预期目标
2. 结构破坏: 关键支撑/阻力被突破（你开仓时设定的关键位）
3. 趋势反转: 多时间框架确认趋势反转
4. 止损接近: 价格接近你的止损位时主动离场
5. 时间止损: 持仓过久但盈亏在小范围徘徊

何时不应该平仓?
1. 正常回调: 趋势中的小幅回撤是正常的，不是反转
2. 市场噪音: 短期波动不代表趋势改变
3. 蚂蚁肉盈利: 微小盈利就跑只会亏交易费
4. 浮盈回撤: 正常的利润回调
5. 结构完好: 你的止损位未被触及，趋势仍在延续
6. "价格接近阻力位": 你开仓时已经考虑了阻力位，不要重新计算价格容错空间

利润回吐的心理陷阱:
- 账面利润回调，很多人会恐慌平仓
- 但如果趋势结构完好，这只是正常的利润回撤
- 过早平仓 = 丢掉了趋势延续的大部分利润
- 正确做法: 评估结构是否破坏，而非盯着账面数字

止损被触发 vs 主动平仓:
- 止损位 = 结构破坏的价格，到达就该离场
- 主动平仓 = 在止损前发现更好的离场理由
- 不要等止损，如果发现入场逻辑失效，应主动离场
- 但也不要因为小幅波动就频繁进出

决策分析框架:

1. 市场状态评估
   - 识别多时间框架的趋势方向和一致性
   - 确定当前价格在市场结构中的位置（接近支撑?阻力?）
   - 评估技术指标对当前趋势的确认程度
   - 观察成交量与价格运动的关系（放量?缩量?）

2. 入场机会识别
   - 趋势方向明确且价格位置合理构成潜在机会
   - 计算风险收益比（潜在收益应显著超过潜在风险）
   - 判断信号强度（多指标共振强于单一信号）
   - 评估置信度（满足条件越多，置信度越高）

3. 平仓时机判断
   - 目标位是否达到（预期风险收益比实现）
   - 结构是否破坏（关键支撑阻力位被突破）
   - 趋势是否减弱（趋势强度指标变化，价格与指标背离）
   - 时间是否过长（机会成本考量）
   - 关键：区分"正常回调"和"趋势反转"

4. 风险管理
   - 根据ATR和市场波动特征确定价格容错空间
   - 杠杆与账户止损百分比呈正比关系
   - 交易时间框架决定最小合理持仓时长
   - 过快止损意味着被市场噪音震出

5. 决策执行
   - 基于以上分析框架形成判断
   - 根据市场状态动态确定参数（价格、杠杆、止损、置信度）
   - 用逻辑推理解释决策依据
   - 调用交易函数后立即停止，不重复调用

重要提示:
- 系统已提供所有必要的市场数据和技术指标
- 你需要综合分析这些信息做出独立判断
- 每个决策都应该基于明确的逻辑和充分的确认
- 调用交易决策函数后立即停止，不要重复调用"""

    def build_decision_request(
        self,
        action_type: str,
        market_context: str,
        position_context: str,
        rag_context: str
    ) -> str:
        """
        构建决策请求

        Args:
            action_type: 决策类型 (entry/exit)
            market_context: 市场上下文
            position_context: 持仓上下文
            rag_context: RAG历史上下文

        Returns:
            完整的决策请求字符串
        """
        action_desc = {
            'entry': '是否应该开仓(做多或做空)',
            'exit': '是否应该平仓'
        }

        request_parts = [
            f"请分析当前情况，决策: {action_desc.get(action_type, action_type)}",
            "",
            "=" * 50,
            market_context,
            "",
            "=" * 50,
            position_context,
            "",
            "=" * 50,
            rag_context,
            "",
            "=" * 50,
            "",
            "决策流程（重要）:",
            "",
            "第1步 - 识别趋势方向:",
            "查看【多时间框架趋势对齐】部分:",
            "- 如果多数时间框架上涨: 趋势向上",
            "- 如果多数时间框架下跌: 趋势向下",
            "- 如果横盘为主或分歧: 趋势不明",
            "",
            "第2步 - 分析价格位置:",
            "查看【市场结构】部分的摆动高低点:",
            "- 价格接近摆动低点: 支撑位附近，做空风险大",
            "- 价格接近摆动高点: 阻力位附近，做多风险大",
            "- 价格在中间位置: 相对安全区域",
            "",
            "第3步 - 评估入场时机(如果是entry决策):",
            "结合趋势和位置:",
            "",
            "第4步 - 评估平仓时机(如果是exit决策):",
            "判断是否应该平仓:",
            "应该平仓的情况:",
            "- 关键支撑/阻力被突破(结构破坏)",
            "- 多个指标显示趋势反转(如MACD交叉,RSI极端)",
            "- 达到合理的盈利目标(风险收益比实现)",
            "- 持仓时间过长但无进展(机会成本)",
            "",
            "不应该平仓的情况:",
            "- 仅仅因为利润回调  ",
            "- 趋势结构完好,只是正常回调",
            "- 关键位未破,多时间框架仍对齐",
            "- 噪音",
            "",
            "第5步 - 计算风险收益比:",
            "- 到止损位的距离 vs 到目标位的距离",
            "- 位置不佳时,宁可等待",
            "",
            "第6步 - 做出决策并立即停止:",
            "调用一个交易决策函数,然后停止。",
            "重要:调用函数后,决策流程自动结束。",
            "",
            "分析要点:",
            "- 综合评估多时间框架趋势的一致性",
            "- 识别当前价格位置与关键支撑阻力的关系",
            "- 判断技术指标对趋势的确认程度",
            "- 计算风险收益比和合理的止损位置",
            "",
            "【止损计算关键步骤】:",
            "",
            "步骤1 - 识别关键价位:",
            "- 找出关键阻力位（做空）或支撑位（做多）",
            "- 例如：市场结构中的摆动高点/低点",
            "",
            "步骤2 - 计算安全边距（必须执行）:",
            "- 保守：安全边距 = 1.5 × ATR",
            "- 激进：安全边距 = 1.0 × ATR",
            "- 最小：安全边距 ≥ 0.5 × ATR",
            "- 记住：止损位必须在关键位**之外**！",
            "",
            "步骤3 - 确定止损触发价（重要！）:",
            "- 做空: 止损触发价 = 阻力位 + 安全边距",
            "- 做多: 止损触发价 = 支撑位 - 安全边距",
            "- 验证：安全边距是否 ≥ 0.5 × ATR",
            "",
            "步骤4 - 计算价格容错空间:",
            "价格容错空间 = |止损触发价 - 入场价| / 入场价",
            "",
            "步骤5 - 验证方向（必须执行）:",
            "- 做空时: 必须验证 止损触发价 > 入场价",
            "- 做多时: 必须验证 止损触发价 < 入场价",
            "- 如果验证失败，说明计算错误，重新计算",
            "",
            "步骤6 - 验证波动空间:",
            "- 计算ATR占价格比例：ATR_pct = ATR / 入场价",
            "- 验证：价格容错空间 >= 2 × ATR_pct（严格要求）",
            "- 验证：价格容错空间 >= 3 × ATR_pct（保守要求，推荐）",
            "- 如果不满足，说明止损太紧，应加大安全边距或放弃交易",
            "",
            "步骤7 - 动态选择杠杆:",
            "",
            "杠杆选择必须根据具体情况动态调整，综合考虑以下因素：",
            "",
            "1. 市场波动性（ATR占价格比例）",
            "   - 波动剧烈 → 降低杠杆，避免被震出",
            "   - 正常波动 → 中等杠杆",
            "   - 波动小 → 可适当提高杠杆",
            "",
            "2. 信号质量（多时间框架对齐度）",
            "   - 时间框架趋势完全一致 → 高质量信号，可提高杠杆",
            "   - 时间框架部分一致 → 中等质量，中等杠杆",
            "   - 时间框架不一致 → 低质量，降低杠杆或观望",
            "",
            "3. 趋势强度（ADX指标）",
            "   - 强趋势 → 趋势稳定，可提高杠杆",
            "   - 中等趋势 → 中等杠杆",
            "   - 弱趋势/横盘 → 降低杠杆或观望",
            "",
            "4. 价格位置精确度",
            "   - 正好在关键支撑/阻力位反弹/回落 → 位置理想，可提高杠杆",
            "   - 在趋势中段，距离关键位较远 → 中等杠杆",
            "   - 价格位置模糊不清 → 降低杠杆或观望",
            "",
            "5. 市场情绪极端性",
            "   - 极端恐惧或极度贪婪 + 反向信号 → 反转机会，可提高杠杆",
            "   - 情绪中性 → 正常杠杆",
            "",
            "步骤8 - 计算账户止损:",
            "账户止损 = 价格容错空间 × 杠杆",
            "",
            "步骤9 - 最终验证:",
            "- 账户止损必须为负数",
            "- 账户止损范围应该在合理区间内",
            "- 因为账户止损看起来很大就降低杠杆缩小止损是错误的",
            "- 如果因为杠杆高导致账户止损过大，应该降低杠杆而非缩小价格容错空间",
            "",
            "核心理念:",
            "- 止损太紧 = 频繁亏损 = 最终亏更多",
            "- 止损大 = 单次风险大 = 但成功率高，最终盈利",
            "- 阻力位不等于止损价，止损要设在阻力位更远处",
            "- 降低杠杆保持大止损，优于高杠杆紧止损"
        ]

        return "\n".join(request_parts)

    def optimize_token_usage(self, text: str, max_tokens: int) -> str:
        """
        优化文本以适应token限制

        Args:
            text: 原始文本
            max_tokens: 最大token数

        Returns:
            优化后的文本
        """
        # 简单估算: 1 token ≈ 4 字符 (英文) 或 1.5 字符 (中文)
        estimated_tokens = len(text) / 2.5

        if estimated_tokens <= max_tokens:
            return text

        # 需要截断
        target_chars = int(max_tokens * 2.5)
        truncated = text[:target_chars]

        # 在最后一个换行符处截断，避免截断句子
        last_newline = truncated.rfind('\n')
        if last_newline > target_chars * 0.8:  # 如果不会损失太多
            truncated = truncated[:last_newline]

        return truncated + "\n... (内容已截断)"

    def _analyze_trend(self, prices: List[float]) -> Dict[str, Any]:
        """
        分析价格趋势（改进版）

        Returns:
            Dict包含: direction (方向), strength (强度), change_pct (变化百分比)
        """
        if len(prices) < 2:
            return {
                "direction": "未知",
                "strength": "无",
                "change_pct": 0
            }

        # 计算价格变化
        total_change = (prices[-1] - prices[0]) / prices[0] * 100

        # 确定方向（降低门槛：从±1%降到±0.5%）
        if total_change > 0.5:
            direction = "上涨"
        elif total_change < -0.5:
            direction = "下跌"
        else:
            direction = "横盘"

        # 计算趋势强度（基于幅度）
        abs_change = abs(total_change)
        if abs_change > 3:
            strength = "强势"
        elif abs_change > 1:
            strength = "中等"
        elif abs_change > 0.5:
            strength = "弱势"
        else:
            strength = "极弱"

        # 计算趋势一致性（价格是否朝同一方向移动）
        changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        if direction != "横盘":
            same_direction = sum(1 for c in changes if (c > 0 and direction == "上涨") or (c < 0 and direction == "下跌"))
            consistency = same_direction / len(changes) * 100
        else:
            consistency = 0

        return {
            "direction": direction,
            "strength": strength,
            "change_pct": total_change,
            "consistency": consistency
        }

    def _format_duration(self, start_time: datetime) -> str:
        """格式化持续时间"""
        if not start_time:
            return "未知"

        from datetime import timezone, datetime as dt
        # freqtrade使用naive UTC时间，根据是否有tzinfo选择对应的now
        now = dt.utcnow() if start_time.tzinfo is None else dt.now(timezone.utc)
        duration = now - start_time
        hours = duration.total_seconds() / 3600

        if hours < 1:
            return f"{int(hours * 60)}分钟"
        elif hours < 24:
            return f"{hours:.1f}小时"
        else:
            return f"{hours / 24:.1f}天"

    def _analyze_market_structure(self, dataframe: pd.DataFrame, lookback: int = 100) -> Dict[str, Any]:
        """分析市场结构（支撑/阻力/趋势）- lookback=100根K线(25小时)能找到更长期的支撑阻力"""
        if len(dataframe) < lookback:
            return {"structure": "数据不足"}

        recent = dataframe.tail(lookback)

        # 计算摆动高低点
        highs = recent['high']
        lows = recent['low']
        closes = recent['close']

        swing_high = highs.max()
        swing_low = lows.min()
        current_price = closes.iloc[-1]

        # 判断市场结构
        # 检查是否在创新高/新低
        prev_highs = dataframe['high'].tail(lookback*2).head(lookback)
        prev_lows = dataframe['low'].tail(lookback*2).head(lookback)

        is_higher_high = swing_high > prev_highs.max() if len(prev_highs) > 0 else False
        is_lower_low = swing_low < prev_lows.min() if len(prev_lows) > 0 else False

        # 确定结构
        if is_higher_high and not is_lower_low:
            structure = "上升结构（Higher Highs）"
        elif is_lower_low and not is_higher_high:
            structure = "下降结构（Lower Lows）"
        elif is_higher_high and is_lower_low:
            structure = "扩张结构（震荡加剧）"
        else:
            structure = "盘整结构（震荡）"

        # 距离关键位的百分比
        distance_to_high = ((swing_high - current_price) / current_price) * 100
        distance_to_low = ((current_price - swing_low) / current_price) * 100

        return {
            "structure": structure,
            "swing_high": swing_high,
            "swing_low": swing_low,
            "distance_to_high_pct": distance_to_high,
            "distance_to_low_pct": distance_to_low,
            "range_pct": ((swing_high - swing_low) / swing_low) * 100
        }

    def _analyze_indicator_trends(self, dataframe: pd.DataFrame) -> Dict[str, str]:
        """分析指标变化趋势"""
        if len(dataframe) < 5:
            return {}

        latest = dataframe.iloc[-1]
        prev = dataframe.iloc[-2]
        prev_5 = dataframe.iloc[-5]

        trends = {}

        # EMA交叉状态
        if 'ema_20' in latest and 'ema_50' in latest:
            ema20_now = latest['ema_20']
            ema50_now = latest['ema_50']
            ema20_prev = prev['ema_20']
            ema50_prev = prev['ema_50']

            if ema20_now > ema50_now and ema20_prev <= ema50_prev:
                trends['ema_cross'] = "刚刚金叉"
            elif ema20_now < ema50_now and ema20_prev >= ema50_prev:
                trends['ema_cross'] = "刚刚死叉"
            elif ema20_now > ema50_now:
                trends['ema_cross'] = "金叉持续中"
            else:
                trends['ema_cross'] = "死叉持续中"

        # MACD柱状图趋势
        if 'macd_hist' in latest:
            macd_hist_now = latest['macd_hist']
            macd_hist_prev = prev['macd_hist']

            if macd_hist_now > macd_hist_prev:
                trends['macd_histogram'] = "增强（多头）" if macd_hist_now > 0 else "减弱（空头弱化）"
            else:
                trends['macd_histogram'] = "减弱（多头弱化）" if macd_hist_now > 0 else "增强（空头）"

        # RSI趋势
        if 'rsi' in latest:
            rsi_now = latest['rsi']
            rsi_5ago = prev_5['rsi']

            if rsi_now > rsi_5ago + 5:
                trends['rsi_trend'] = "上升（动能增强）"
            elif rsi_now < rsi_5ago - 5:
                trends['rsi_trend'] = "下降（动能减弱）"
            else:
                trends['rsi_trend'] = "平稳"

        # ADX趋势强度
        if 'adx' in latest:
            adx_now = latest['adx']
            adx_prev = prev['adx']

            if adx_now > adx_prev:
                trends['adx_direction'] = f"上升（趋势增强，当前{adx_now:.1f}）"
            else:
                trends['adx_direction'] = f"下降（趋势减弱，当前{adx_now:.1f}）"

        return trends

    def _analyze_timeframe_alignment(self, dataframe: pd.DataFrame) -> Dict[str, Any]:
        """
        分析多时间框架趋势对齐（改进版，支持横盘）
        """
        if len(dataframe) < 2:
            return {"alignment": "数据不足", "trends": {}}

        latest = dataframe.iloc[-1]
        trends = {}

        # 横盘判断阈值：EMA差距小于0.3%视为横盘
        consolidation_threshold = 0.003

        # 15分钟趋势
        if 'ema_20' in latest and 'ema_50' in latest:
            ema_diff = (latest['ema_20'] - latest['ema_50']) / latest['ema_50']
            if abs(ema_diff) < consolidation_threshold:
                trends['15m'] = "横盘"
            elif ema_diff > 0:
                trends['15m'] = "上涨"
            else:
                trends['15m'] = "下跌"

        # 1小时趋势
        if 'ema_20_1h' in latest and 'ema_50_1h' in latest:
            ema_diff = (latest['ema_20_1h'] - latest['ema_50_1h']) / latest['ema_50_1h']
            if abs(ema_diff) < consolidation_threshold:
                trends['1h'] = "横盘"
            elif ema_diff > 0:
                trends['1h'] = "上涨"
            else:
                trends['1h'] = "下跌"

        # 4小时趋势
        if 'ema_20_4h' in latest and 'ema_50_4h' in latest:
            ema_diff = (latest['ema_20_4h'] - latest['ema_50_4h']) / latest['ema_50_4h']
            if abs(ema_diff) < consolidation_threshold:
                trends['4h'] = "横盘"
            elif ema_diff > 0:
                trends['4h'] = "上涨"
            else:
                trends['4h'] = "下跌"

        # 1天趋势
        if 'ema_20_1d' in latest and 'ema_50_1d' in latest:
            ema_diff = (latest['ema_20_1d'] - latest['ema_50_1d']) / latest['ema_50_1d']
            if abs(ema_diff) < consolidation_threshold:
                trends['1d'] = "横盘"
            elif ema_diff > 0:
                trends['1d'] = "上涨"
            else:
                trends['1d'] = "下跌"

        # 判断对齐情况
        if not trends:
            return {"alignment": "无趋势数据", "trends": {}}

        uptrend_count = sum(1 for t in trends.values() if t == "上涨")
        downtrend_count = sum(1 for t in trends.values() if t == "下跌")
        consolidation_count = sum(1 for t in trends.values() if t == "横盘")
        total_count = len(trends)

        if uptrend_count == total_count:
            alignment = "完全对齐 - 强势上涨"
        elif downtrend_count == total_count:
            alignment = "完全对齐 - 强势下跌"
        elif consolidation_count == total_count:
            alignment = "完全对齐 - 横盘整理"
        elif consolidation_count >= total_count / 2:
            alignment = f"横盘为主（{consolidation_count}/{total_count}）- 等待突破"
        elif uptrend_count > downtrend_count:
            alignment = f"多数上涨（{uptrend_count}/{total_count}）"
        elif downtrend_count > uptrend_count:
            alignment = f"多数下跌（{downtrend_count}/{total_count}）"
        else:
            alignment = "趋势分歧（震荡）"

        return {
            "trends": trends,
            "alignment": alignment,
            "strength": uptrend_count / total_count if uptrend_count > downtrend_count else -downtrend_count / total_count
        }

    def _analyze_volume_trend(self, dataframe: pd.DataFrame, lookback: int = 100) -> Dict[str, Any]:
        """分析成交量趋势 - lookback=100根K线(25小时)能更好识别放量/缩量趋势"""
        if len(dataframe) < lookback:
            return {"trend": "数据不足"}

        recent = dataframe.tail(lookback)
        volumes = recent['volume']

        # 计算成交量移动平均
        volume_ma = volumes.mean()
        current_volume = volumes.iloc[-1]

        # 成交量趋势
        first_half_avg = volumes.head(lookback//2).mean()
        second_half_avg = volumes.tail(lookback//2).mean()

        if second_half_avg > first_half_avg * 1.2:
            trend = "持续放量"
        elif second_half_avg < first_half_avg * 0.8:
            trend = "持续缩量"
        else:
            trend = "平稳"

        # 当前成交量相对于平均值
        volume_ratio = current_volume / volume_ma

        if volume_ratio > 1.5:
            current_status = "异常放量"
        elif volume_ratio > 1.2:
            current_status = "明显放量"
        elif volume_ratio < 0.7:
            current_status = "明显缩量"
        else:
            current_status = "正常"

        return {
            "trend": trend,
            "current_status": current_status,
            "volume_ratio": volume_ratio,
            "current_vs_avg": f"{(volume_ratio - 1) * 100:+.1f}%"
        }

    def _get_indicator_history(self, dataframe: pd.DataFrame, lookback: int = 100) -> Dict[str, List]:
        """获取关键指标的历史序列 - 从lookback根中取最近20根展示"""
        if len(dataframe) < lookback:
            lookback = len(dataframe)

        recent = dataframe.tail(lookback)
        history = {}

        # 选择关键指标提供历史序列
        key_indicators = ['close', 'rsi', 'macd', 'macd_hist', 'ema_20', 'ema_50', 'adx', 'volume']

        for ind in key_indicators:
            if ind in recent.columns:
                values = recent[ind].tolist()
                # 只保留最近的值，格式化为简洁形式
                history[f"{ind}_recent"] = [round(float(v), 2) if pd.notna(v) else None for v in values[-20:]]

        return history
