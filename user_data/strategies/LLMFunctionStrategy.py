"""
LLM Function Calling Strategy
基于LLM函数调用和RAG的智能交易策略

作者: Claude Code
版本: 1.0.0
"""

import logging
from typing import Dict, Any, Optional
import pandas as pd
from datetime import datetime
from freqtrade.strategy import IStrategy, informative, merge_informative_pair
import talib.abstract as ta

# 导入自定义模块
from llm_modules.utils.config_loader import ConfigLoader
from llm_modules.utils.context_builder import ContextBuilder
from llm_modules.tools.trading_tools import TradingTools
from llm_modules.llm.llm_client import LLMClient
from llm_modules.llm.function_executor import FunctionExecutor
from llm_modules.experience.trade_logger import TradeLogger
from llm_modules.experience.experience_manager import ExperienceManager

# 初始化 logger（必须在使用前定义）
logger = logging.getLogger(__name__)

# RAG系统相关导入（可选）
try:
    from llm_modules.rag.embedding_service import EmbeddingService
    from llm_modules.rag.vector_store import VectorStore
    from llm_modules.rag.rag_manager import RAGManager
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    logger.warning("RAG模块不可用，将在简化模式下运行")


class LLMFunctionStrategy(IStrategy):
    """
    LLM函数调用策略

    特性:
    - OpenAI Function Calling 完整交易控制
    - RAG语义检索系统 (text-embedding-bge-m3)
    - 支持期货、多空双向、动态杠杆
    - 经验学习和持续优化
    """

    # 策略基本配置
    INTERFACE_VERSION = 3
    can_short = True
    timeframe = '15m'  # 15分钟K线，适合中短线趋势

    # 启动需要的历史数据
    startup_candle_count = 800  # 15分钟*800 = 约8.3天数据（确保4小时框架EMA50稳定）

    # 止损配置
    stoploss = -0.99  # 初始止损99%，将由LLM的custom_stoploss动态覆盖
    use_custom_stoploss = True  # 启用自定义止损

    # 仓位调整
    position_adjustment_enable = True
    max_entry_position_adjustment = 10

    # 订单类型
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False,
    }

    def __init__(self, config: dict) -> None:
        """初始化策略"""
        super().__init__(config)

        logger.info("=" * 60)
        logger.info("LLM Function Calling Strategy - 正在初始化...")
        logger.info("=" * 60)

        try:
            # 1. 加载配置
            self.config_loader = ConfigLoader()
            self.llm_config = self.config_loader.get_llm_config()
            self.rag_config = self.config_loader.get_rag_config()
            self.risk_config = self.config_loader.get_risk_config()
            self.experience_config = self.config_loader.get_experience_config()
            self.context_config = self.config_loader.get_context_config()

            # 2. 初始化工具类
            self.context_builder = ContextBuilder(self.context_config)

            # 3. 初始化函数执行器
            self.function_executor = FunctionExecutor()

            # 4. 初始化交易工具（简化版 - 只保留交易控制工具）
            self.trading_tools = TradingTools(self)

            # 5. 初始化RAG系统（可选）
            self.rag_manager = None
            if RAG_AVAILABLE and self.config_loader.is_rag_enabled():
                try:
                    embedding_service = EmbeddingService(self.llm_config)
                    vector_store = VectorStore(
                        storage_path=self.rag_config.get("storage_path", "./user_data/data/vector_store"),
                        embedding_service=embedding_service
                    )
                    self.rag_manager = RAGManager(
                        rag_config=self.rag_config,
                        embedding_service=embedding_service,
                        vector_store=vector_store
                    )
                    logger.info("✓ RAG系统已启用")
                except Exception as e:
                    logger.error(f"✗ RAG系统初始化失败: {e}，将继续以简化模式运行")
                    self.rag_manager = None
            else:
                logger.info("✓ RAG系统已禁用，使用简化模式")

            # 6. 初始化LLM客户端
            self.llm_client = LLMClient(self.llm_config, self.function_executor)

            # 7. 注册所有工具函数
            self._register_all_tools()

            # 8. 初始化经验系统
            self.trade_logger = TradeLogger(self.experience_config)
            self.experience_manager = ExperienceManager(
                trade_logger=self.trade_logger,
                rag_manager=self.rag_manager
            )

            # 10. 缓存
            self._leverage_cache = {}
            self._stoploss_cache = {}
            self._position_adjustment_cache = {}
            self._entry_price_cache = {}
            self._exit_price_cache = {}

            # 11. 系统提示词
            self.system_prompt = self.context_builder.build_system_prompt()

            logger.info("✓ 策略初始化完成")
            logger.info(f"  - LLM模型: {self.llm_config.get('model')}")
            logger.info(f"  - 交易工具已注册: {len(self.function_executor.list_functions())} 个")
            logger.info(f"  - RAG系统: {'启用' if self.rag_manager else '禁用'}")
            logger.info(f"  - 模式: 简化版（市场数据已内置在context中）")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"策略初始化失败: {e}", exc_info=True)
            raise

    def _register_all_tools(self):
        """注册所有工具函数（简化版 - 只注册交易控制工具）"""
        # 只注册交易工具（市场数据、账户信息已在context中提供）
        if self.trading_tools:
            self.function_executor.register_tools_from_instance(
                self.trading_tools,
                self.trading_tools.get_tools_schema()
            )
            logger.debug(f"已注册 {len(self.trading_tools.get_tools_schema())} 个交易控制函数")

    def bot_start(self, **kwargs) -> None:
        """
        策略启动时调用（此时dp和wallets已初始化）
        """
        logger.info("✓ Bot已启动，策略运行中...")
        logger.info(f"✓ 交易工具: {len(self.function_executor.list_functions())} 个函数可用")

    # 多时间框架数据支持
    @informative('1h')
    def populate_indicators_1h(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """1小时数据指标"""
        dataframe['ema_20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macd_signal'] = macd['macdsignal']
        bollinger = ta.BBANDS(dataframe, timeperiod=20)
        dataframe['bb_upper'] = bollinger['upperband']
        dataframe['bb_lower'] = bollinger['lowerband']
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        return dataframe

    @informative('4h')
    def populate_indicators_4h(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """4小时数据指标"""
        dataframe['ema_20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macd_signal'] = macd['macdsignal']
        bollinger = ta.BBANDS(dataframe, timeperiod=20)
        dataframe['bb_upper'] = bollinger['upperband']
        dataframe['bb_lower'] = bollinger['lowerband']
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        return dataframe

    @informative('1d')
    def populate_indicators_1d(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """日线数据指标（注意：8天数据只有8根日线K线，EMA50勉强可用，已删除EMA200）"""
        dataframe['ema_20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        # dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)  # 需要200天数据，删除
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macd_signal'] = macd['macdsignal']
        bollinger = ta.BBANDS(dataframe, timeperiod=20)
        dataframe['bb_upper'] = bollinger['upperband']
        dataframe['bb_lower'] = bollinger['lowerband']
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        return dataframe

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        计算技术指标（15分钟基础数据）
        """
        # 趋势指标
        dataframe['ema_20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_100'] = ta.EMA(dataframe, timeperiod=100)  # 用EMA100代替EMA200，更适合15分钟框架

        # 动量指标
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macd_signal'] = macd['macdsignal']
        dataframe['macd_hist'] = macd['macdhist']

        # 波动率指标
        bollinger = ta.BBANDS(dataframe, timeperiod=20)
        dataframe['bb_upper'] = bollinger['upperband']
        dataframe['bb_middle'] = bollinger['middleband']
        dataframe['bb_lower'] = bollinger['lowerband']

        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)

        # 趋势强度
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)

        # 成交量指标
        dataframe['mfi'] = ta.MFI(dataframe, timeperiod=14)
        dataframe['obv'] = ta.OBV(dataframe)

        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        开仓信号 - 由LLM决策
        """
        pair = metadata['pair']

        # 默认不开仓
        dataframe.loc[:, 'enter_long'] = 0
        dataframe.loc[:, 'enter_short'] = 0
        dataframe.loc[:, 'enter_tag'] = ''

        # 只在最新的K线上做决策
        if len(dataframe) < self.startup_candle_count:
            return dataframe

        try:
            # 获取当前所有持仓（用于传给context_builder）
            from freqtrade.persistence import Trade
            current_trades = Trade.get_open_trades()

            # 构建完整的市场上下文（包含技术指标、账户信息、持仓情况）
            # 获取exchange对象用于市场情绪数据
            exchange = None
            if hasattr(self, 'dp') and self.dp:
                if hasattr(self.dp, '_exchange'):
                    exchange = self.dp._exchange
                elif hasattr(self.dp, 'exchange'):
                    exchange = self.dp.exchange

            market_context = self.context_builder.build_market_context(
                dataframe=dataframe,
                metadata=metadata,
                wallets=self.wallets,
                current_trades=current_trades,
                exchange=exchange
            )

            # 检索相似的历史情况（如果RAG可用）
            rag_context = ""
            if self.rag_manager:
                try:
                    rag_context = self.rag_manager.get_relevant_context(
                        pair=pair,
                        current_state=market_context,
                        action_type="entry"
                    )
                except Exception as e:
                    logger.warning(f"RAG检索失败: {e}")
                    rag_context = ""

            # 构建决策请求
            decision_request = self.context_builder.build_decision_request(
                action_type="entry",
                market_context=market_context,
                position_context="",  # 已包含在market_context中
                rag_context=rag_context
            )

            # 调用LLM决策
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": decision_request}
            ]

            response = self.llm_client.call_with_functions(
                messages=messages,
                max_iterations=10  # 限制迭代次数，防止无限循环
            )

            # 处理响应
            if response.get("success"):
                function_calls = response.get("function_calls", [])
                llm_message = response.get("message", "")

                # 检查是否有交易信号
                signal = self.trading_tools.get_signal(pair)

                # 提取置信度用于记录决策
                confidence = signal.get("confidence_score", 50) / 100 if signal else 0.5

                # 记录决策
                self.experience_manager.log_decision_with_context(
                    pair=pair,
                    action="entry",
                    decision=llm_message,
                    reasoning=str(function_calls),
                    confidence=confidence,
                    market_context={"indicators": market_context},
                    function_calls=function_calls
                )

                if signal:
                    action = signal.get("action")
                    reason = signal.get("reason", llm_message)
                    limit_price = signal.get("limit_price")

                    # 提取新增参数
                    confidence_score = signal.get("confidence_score", 0)
                    key_support = signal.get("key_support", 0)
                    key_resistance = signal.get("key_resistance", 0)
                    rsi_value = signal.get("rsi_value", 0)
                    trend_strength = signal.get("trend_strength", "未知")

                    if action == "enter_long":
                        # 缓存挂单价格
                        if limit_price:
                            self._entry_price_cache[pair] = limit_price
                        dataframe.loc[dataframe.index[-1], 'enter_long'] = 1
                        dataframe.loc[dataframe.index[-1], 'enter_tag'] = reason
                        logger.info(f"📈 {pair} | 做多 | 置信度: {confidence_score}")
                        logger.info(f"   挂单价: {limit_price} | 支撑: {key_support} | 阻力: {key_resistance}")
                        logger.info(f"   RSI: {rsi_value} | 趋势强度: {trend_strength}")
                        logger.info(f"   理由: {reason}")
                    elif action == "enter_short":
                        # 缓存挂单价格
                        if limit_price:
                            self._entry_price_cache[pair] = limit_price
                        dataframe.loc[dataframe.index[-1], 'enter_short'] = 1
                        dataframe.loc[dataframe.index[-1], 'enter_tag'] = reason
                        logger.info(f"📉 {pair} | 做空 | 置信度: {confidence_score}")
                        logger.info(f"   挂单价: {limit_price} | 支撑: {key_support} | 阻力: {key_resistance}")
                        logger.info(f"   RSI: {rsi_value} | 趋势强度: {trend_strength}")
                        logger.info(f"   理由: {reason}")
                    elif action == "hold":
                        logger.info(f"🔒 {pair} | 保持持仓 | 置信度: {confidence_score} | RSI: {rsi_value}")
                        logger.info(f"   理由: {reason}")
                    elif action == "wait":
                        logger.info(f"⏸️  {pair} | 空仓等待 | 置信度: {confidence_score} | RSI: {rsi_value}")
                        logger.info(f"   理由: {reason}")
                else:
                    # 没有交易信号 = 观望，显示LLM的完整分析
                    logger.info(f"⏸️  {pair} | 未提供明确信号\n{llm_message}")

                # 清空信号缓存
                self.trading_tools.clear_signals()

        except Exception as e:
            logger.error(f"开仓决策失败 {pair}: {e}")

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        平仓信号 - 由LLM决策
        """
        pair = metadata['pair']

        # 默认不平仓
        dataframe.loc[:, 'exit_long'] = 0
        dataframe.loc[:, 'exit_short'] = 0
        dataframe.loc[:, 'exit_tag'] = ''

        # 只在最新的K线上做决策
        if len(dataframe) < self.startup_candle_count:
            return dataframe

        try:
            # 获取当前所有持仓
            from freqtrade.persistence import Trade
            current_trades = Trade.get_open_trades()

            # 检查当前交易对是否有持仓
            pair_has_position = any(t.pair == pair for t in current_trades)
            if not pair_has_position:
                return dataframe  # 无持仓，不需要决策

            # 构建完整的市场上下文（包含技术指标、账户信息、持仓情况）
            # 获取exchange对象用于市场情绪数据
            exchange = None
            if hasattr(self, 'dp') and self.dp:
                if hasattr(self.dp, '_exchange'):
                    exchange = self.dp._exchange
                elif hasattr(self.dp, 'exchange'):
                    exchange = self.dp.exchange

            market_context = self.context_builder.build_market_context(
                dataframe=dataframe,
                metadata=metadata,
                wallets=self.wallets,
                current_trades=current_trades,
                exchange=exchange
            )

            # 检索相似的历史情况（如果RAG可用）
            rag_context = ""
            if self.rag_manager:
                try:
                    rag_context = self.rag_manager.get_relevant_context(
                        pair=pair,
                        current_state=market_context,
                        action_type="exit"
                    )
                except Exception as e:
                    logger.warning(f"RAG检索失败: {e}")
                    rag_context = ""

            # 构建决策请求
            decision_request = self.context_builder.build_decision_request(
                action_type="exit",
                market_context=market_context,
                position_context="",  # 已包含在market_context中
                rag_context=rag_context
            )

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": decision_request}
            ]

            response = self.llm_client.call_with_functions(
                messages=messages,
                max_iterations=10  # 限制迭代次数，防止无限循环
            )

            if response.get("success"):
                llm_message = response.get("message", "")
                signal = self.trading_tools.get_signal(pair)
                if signal and signal.get("action") == "exit":
                    reason = signal.get("reason", llm_message)
                    limit_price = signal.get("limit_price")

                    # 提取新增参数
                    confidence_score = signal.get("confidence_score", 0)
                    rsi_value = signal.get("rsi_value", 0)

                    # 缓存挂单价格
                    if limit_price:
                        self._exit_price_cache[pair] = limit_price

                    dataframe.loc[dataframe.index[-1], 'exit_long'] = 1
                    dataframe.loc[dataframe.index[-1], 'exit_short'] = 1
                    dataframe.loc[dataframe.index[-1], 'exit_tag'] = reason
                    logger.info(f"🔚 {pair} | 平仓 | 置信度: {confidence_score} | 挂单价: {limit_price}")
                    logger.info(f"   RSI: {rsi_value}")
                    logger.info(f"   理由: {reason}")
                else:
                    logger.info(f"💎 {pair} | 继续持有\n{llm_message}")

                self.trading_tools.clear_signals()

        except Exception as e:
            logger.error(f"平仓决策失败 {pair}: {e}")

        return dataframe

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs
    ) -> float:
        """
        动态杠杆 - 由LLM决定或使用缓存值
        """
        # 检查缓存
        if pair in self._leverage_cache:
            leverage_value = self._leverage_cache[pair]
            del self._leverage_cache[pair]  # 使用后清除
            return min(leverage_value, max_leverage)

        # 默认杠杆
        default_leverage = self.risk_config.get("default_leverage", 10)
        return min(default_leverage, max_leverage)

    def custom_stoploss(
        self,
        pair: str,
        trade: Any,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs
    ) -> Optional[float]:
        """
        动态止损 - 由LLM完全控制

        重要说明：
        - current_profit 是当前仓位的盈亏百分比(已考虑杠杆)
        - 返回值是相对于当前价格的止损距离(负数)
        - 当 current_profit <= 返回值 时触发止损

        示例：
        - AI设置: -20%账户止损
        - current_profit = -0.20 时触发
        - 直接返回 -0.20 即可
        """
        logger.debug(f"custom_stoploss被调用: {pair}, profit={current_profit*100:.2f}%, cache={pair in self._stoploss_cache}")

        # 检查缓存中是否有LLM设置的止损
        if pair in self._stoploss_cache:
            # LLM设置的账户止损百分比（例如 -20 表示账户亏损20%）
            stoploss_pct = self._stoploss_cache[pair]

            # 转换为小数形式（-20 -> -0.20）
            stoploss_value = stoploss_pct / 100

            # 获取杠杆和开仓价用于日志
            leverage = getattr(trade, 'leverage', 1)
            open_rate = trade.open_rate

            # 计算触发止损时的价格（用于日志显示）
            # current_profit = (current_rate - open_rate) / open_rate * leverage (做多)
            # current_profit = (open_rate - current_rate) / open_rate * leverage (做空)
            # 反推: price_change = stoploss_value / leverage
            price_change_pct = stoploss_value / leverage

            if trade.is_short:
                # 做空：价格上涨触发止损
                stoploss_price = open_rate * (1 - price_change_pct)
            else:
                # 做多：价格下跌触发止损
                stoploss_price = open_rate * (1 + price_change_pct)

            # 止损信息记录到 debug 级别
            logger.debug(f"止损: {pair} {stoploss_pct}% @ {stoploss_price:.2f} (杠杆{leverage}x)")
            logger.debug(f"  开仓价: {open_rate:.2f}, 当前价: {current_rate:.2f}, 当前盈亏: {current_profit*100:.2f}%")
            logger.debug(f"  方向: {'做空' if trade.is_short else '做多'}, 触发条件: 盈亏 <= {stoploss_value*100:.2f}%")
            
            # 安全检查：确保止损值在合理范围内
            if stoploss_value > 0:
                logger.error(f"❌ {pair} 止损值异常: {stoploss_value} 应该是负数!")
                return -0.10  # 返回保守的10%止损
            
            if stoploss_value < -0.99:
                logger.warning(f"⚠️ {pair} 止损值过大: {stoploss_value}, 限制为-99%")
                return -0.99
            
            return stoploss_value
        
        # 如果没有LLM设置的止损，使用保守的默认值
        default_stoploss = -0.10  # 默认账户亏损10%时止损
        logger.debug(f"{pair} 使用默认止损: {default_stoploss*100:.2f}%")
        return default_stoploss

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: Optional[float],
        max_stake: float,
        leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs
    ) -> float:
        """
        动态仓位大小 - 可由LLM调整
        """
        # 使用默认的stake amount
        return proposed_stake

    def custom_entry_price(
        self,
        pair: str,
        current_time: datetime,
        proposed_rate: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs
    ) -> float:
        """
        自定义入场价格 - 允许LLM指定挂单价格

        Args:
            pair: 交易对
            proposed_rate: freqtrade建议的价格
            其他参数...

        Returns:
            入场价格
        """
        # 检查LLM是否指定了挂单价格
        if pair in self._entry_price_cache:
            price = self._entry_price_cache[pair]
            del self._entry_price_cache[pair]
            logger.info(f"{pair} 使用LLM指定的入场价格: {price}")
            return price

        # 使用默认价格
        return proposed_rate

    def custom_exit_price(
        self,
        pair: str,
        trade: Any,
        current_time: datetime,
        proposed_rate: float,
        current_profit: float,
        exit_tag: Optional[str],
        **kwargs
    ) -> float:
        """
        自定义出场价格 - 允许LLM指定挂单价格

        Args:
            pair: 交易对
            proposed_rate: freqtrade建议的价格
            其他参数...

        Returns:
            出场价格
        """
        # 检查LLM是否指定了挂单价格
        if pair in self._exit_price_cache:
            price = self._exit_price_cache[pair]
            del self._exit_price_cache[pair]
            logger.info(f"{pair} 使用LLM指定的出场价格: {price}")
            return price

        # 使用默认价格
        return proposed_rate

    def adjust_trade_position(
        self,
        trade: Any,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: Optional[float],
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs
    ) -> Optional[float]:
        """
        仓位调整 - 允许LLM加仓或减仓

        Args:
            trade: 当前交易对象
            current_rate: 当前价格
            其他参数...

        Returns:
            Optional[float]: 要增加的stake金额（正数=加仓，负数=减仓），None=不调整
        """
        pair = trade.pair

        # 检查LLM是否有仓位调整决策
        if pair in self._position_adjustment_cache:
            adjustment_info = self._position_adjustment_cache[pair]
            del self._position_adjustment_cache[pair]

            adjustment_pct = adjustment_info.get("adjustment_pct", 0)
            limit_price = adjustment_info.get("limit_price")
            reason = adjustment_info.get("reason", "")

            # 计算调整金额
            current_stake = trade.stake_amount
            adjustment_stake = current_stake * (adjustment_pct / 100)

            if adjustment_pct > 0:
                # 加仓
                adjustment_stake = min(adjustment_stake, max_stake)
                if min_stake and adjustment_stake < min_stake:
                    logger.warning(f"{pair} 加仓金额 {adjustment_stake} 低于最小stake {min_stake}")
                    return None

                # 缓存挂单价格（用于加仓订单）
                if limit_price:
                    self._entry_price_cache[pair] = limit_price

                logger.info(f"{pair} 加仓 {adjustment_pct:.1f}% = {adjustment_stake:.2f} USDT | 挂单价: {limit_price} | {reason}")
                return adjustment_stake

            elif adjustment_pct < 0:
                # 减仓
                max_reduce = -current_stake * 0.99  # 最多减99%（保留一点避免完全平仓）
                adjustment_stake = max(adjustment_stake, max_reduce)

                # 缓存挂单价格（用于减仓订单）
                if limit_price:
                    self._exit_price_cache[pair] = limit_price

                logger.info(f"{pair} 减仓 {abs(adjustment_pct):.1f}% = {adjustment_stake:.2f} USDT | 挂单价: {limit_price} | {reason}")
                return adjustment_stake

        # 无调整
        return None
