"""
交易控制工具模块（简化版）
提供LLM可调用的6个核心交易操作
"""
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class TradingTools:
    """交易控制工具集（简化版）"""

    def __init__(self, strategy_instance):
        """
        初始化交易工具

        Args:
            strategy_instance: freqtrade策略实例
        """
        self.strategy = strategy_instance
        self._signal_cache = {}  # 缓存本周期的信号

    def get_tools_schema(self) -> list[Dict[str, Any]]:
        """获取所有交易工具的OpenAI函数schema"""
        return [
            {
                "name": "signal_entry_long",
                "description": "开多仓 - 做多开仓并指定挂单价格、杠杆、止损",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pair": {
                            "type": "string",
                            "description": "交易对，例如 BTC/USDT:USDT"
                        },
                        "limit_price": {
                            "type": "number",
                            "description": "挂单价格 - 你期望的入场价格（通常略低于当前价以获得更好成交价）"
                        },
                        "leverage": {
                            "type": "number",
                            "description": "杠杆倍数 (1-100)"
                        },
                        "stoploss_pct": {
                            "type": "number",
                            "description": "止损百分比（负数），例如 -10 表示账户亏损10%时止损"
                        },
                        "confidence_score": {
                            "type": "number",
                            "description": "决策置信度 (1-100)，表示你对这个决策的信心程度。>80高信心，60-80中等，<60低信心"
                        },
                        "key_support": {
                            "type": "number",
                            "description": "关键支撑位价格 - 下方重要支撑价格，建议止损设在此价格下方"
                        },
                        "key_resistance": {
                            "type": "number",
                            "description": "关键阻力位价格 - 上方重要阻力价格，可作为目标价"
                        },
                        "rsi_value": {
                            "type": "number",
                            "description": "当前RSI数值 (0-100)"
                        },
                        "trend_strength": {
                            "type": "string",
                            "description": "趋势强度评估: '强势' | '中等' | '弱势'"
                        },
                        "reason": {
                            "type": "string",
                            "description": "开仓理由 - 说明为什么做多，包括技术面、趋势判断等"
                        }
                    },
                    "required": ["pair", "limit_price", "leverage", "stoploss_pct", "confidence_score", "key_support", "key_resistance", "rsi_value", "trend_strength", "reason"]
                }
            },
            {
                "name": "signal_entry_short",
                "description": "开空仓 - 做空开仓并指定挂单价格、杠杆、止损",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pair": {
                            "type": "string",
                            "description": "交易对"
                        },
                        "limit_price": {
                            "type": "number",
                            "description": "挂单价格 - 你期望的入场价格（通常略高于当前价以获得更好成交价）"
                        },
                        "leverage": {
                            "type": "number",
                            "description": "杠杆倍数 (1-100)"
                        },
                        "stoploss_pct": {
                            "type": "number",
                            "description": "止损百分比（负数），例如 -10 表示账户亏损10%时止损"
                        },
                        "confidence_score": {
                            "type": "number",
                            "description": "决策置信度 (1-100)，表示你对这个决策的信心程度。>80高信心，60-80中等，<60低信心"
                        },
                        "key_support": {
                            "type": "number",
                            "description": "关键支撑位价格 - 下方重要支撑价格，可作为目标价"
                        },
                        "key_resistance": {
                            "type": "number",
                            "description": "关键阻力位价格 - 上方重要阻力价格，建议止损设在此价格上方"
                        },
                        "rsi_value": {
                            "type": "number",
                            "description": "当前RSI数值 (0-100)"
                        },
                        "trend_strength": {
                            "type": "string",
                            "description": "趋势强度评估: '强势' | '中等' | '弱势'"
                        },
                        "reason": {
                            "type": "string",
                            "description": "开仓理由"
                        }
                    },
                    "required": ["pair", "limit_price", "leverage", "stoploss_pct", "confidence_score", "key_support", "key_resistance", "rsi_value", "trend_strength", "reason"]
                }
            },
            {
                "name": "signal_exit",
                "description": "平仓 - 平掉当前持仓并指定挂单价格",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pair": {
                            "type": "string",
                            "description": "交易对"
                        },
                        "limit_price": {
                            "type": "number",
                            "description": "挂单价格 - 你期望的出场价格"
                        },
                        "confidence_score": {
                            "type": "number",
                            "description": "决策置信度 (1-100)"
                        },
                        "rsi_value": {
                            "type": "number",
                            "description": "当前RSI数值 (0-100)"
                        },
                        "reason": {
                            "type": "string",
                            "description": "平仓理由 - 说明为什么平仓"
                        }
                    },
                    "required": ["pair", "limit_price", "confidence_score", "rsi_value", "reason"]
                }
            },
            {
                "name": "adjust_position",
                "description": "加仓/减仓 - 调整现有持仓大小并指定挂单价格",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pair": {
                            "type": "string",
                            "description": "交易对"
                        },
                        "adjustment_pct": {
                            "type": "number",
                            "description": "调整百分比 (正数=加仓, 负数=减仓)，例如 50 表示加仓50%，-30 表示减仓30%"
                        },
                        "limit_price": {
                            "type": "number",
                            "description": "挂单价格 - 你期望的加仓/减仓价格"
                        },
                        "confidence_score": {
                            "type": "number",
                            "description": "决策置信度 (1-100)"
                        },
                        "key_support": {
                            "type": "number",
                            "description": "关键支撑位价格"
                        },
                        "key_resistance": {
                            "type": "number",
                            "description": "关键阻力位价格"
                        },
                        "reason": {
                            "type": "string",
                            "description": "调整理由 - 说明为什么加仓或减仓"
                        }
                    },
                    "required": ["pair", "adjustment_pct", "limit_price", "confidence_score", "key_support", "key_resistance", "reason"]
                }
            },
            {
                "name": "signal_hold",
                "description": "保持 - 持仓不动，维持当前仓位（用于已有仓位时）",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pair": {
                            "type": "string",
                            "description": "交易对"
                        },
                        "confidence_score": {
                            "type": "number",
                            "description": "决策置信度 (1-100) - 表示继续持有的信心"
                        },
                        "rsi_value": {
                            "type": "number",
                            "description": "当前RSI数值 (0-100)"
                        },
                        "reason": {
                            "type": "string",
                            "description": "保持理由 - 说明为什么继续持有"
                        }
                    },
                    "required": ["pair", "confidence_score", "rsi_value", "reason"]
                }
            },
            {
                "name": "signal_wait",
                "description": "等待 - 空仓观望，不进行任何操作（用于无仓位时）",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pair": {
                            "type": "string",
                            "description": "交易对"
                        },
                        "confidence_score": {
                            "type": "number",
                            "description": "决策置信度 (1-100) - 表示不开仓的信心（信心低说明可能有机会但不确定）"
                        },
                        "rsi_value": {
                            "type": "number",
                            "description": "当前RSI数值 (0-100)"
                        },
                        "reason": {
                            "type": "string",
                            "description": "等待理由 - 说明为什么不开仓"
                        }
                    },
                    "required": ["pair", "confidence_score", "rsi_value", "reason"]
                }
            }
        ]

    def signal_entry_long(
        self,
        pair: str,
        limit_price: float,
        leverage: float,
        stoploss_pct: float,
        confidence_score: float,
        key_support: float,
        key_resistance: float,
        rsi_value: float,
        trend_strength: str,
        reason: str
    ) -> Dict[str, Any]:
        """
        发出做多信号

        Args:
            pair: 交易对
            limit_price: 挂单价格
            leverage: 杠杆倍数
            stoploss_pct: 止损百分比（负数）
            confidence_score: 决策置信度 (1-100)
            key_support: 关键支撑位
            key_resistance: 关键阻力位
            rsi_value: RSI数值
            trend_strength: 趋势强度
            reason: 开仓理由

        Returns:
            执行结果
        """
        try:
            # 验证参数
            if leverage < 1 or leverage > 100:
                return {"success": False, "message": "杠杆必须在1-100之间"}

            if stoploss_pct >= 0:
                return {"success": False, "message": "止损必须为负数"}

            if limit_price <= 0:
                return {"success": False, "message": "挂单价格必须大于0"}

            if confidence_score < 1 or confidence_score > 100:
                return {"success": False, "message": "置信度必须在1-100之间"}

            # 缓存信号（包含所有参数）
            self._signal_cache[pair] = {
                "action": "enter_long",
                "limit_price": limit_price,
                "leverage": leverage,
                "stoploss_pct": stoploss_pct,
                "confidence_score": confidence_score,
                "key_support": key_support,
                "key_resistance": key_resistance,
                "rsi_value": rsi_value,
                "trend_strength": trend_strength,
                "reason": reason
            }

            # 同时设置杠杆和止损到策略缓存
            if not hasattr(self.strategy, '_leverage_cache'):
                self.strategy._leverage_cache = {}
            if not hasattr(self.strategy, '_stoploss_cache'):
                self.strategy._stoploss_cache = {}

            self.strategy._leverage_cache[pair] = leverage
            self.strategy._stoploss_cache[pair] = stoploss_pct

            logger.info(f"[做多信号] {pair} | 置信度: {confidence_score} | 挂单价: {limit_price} | 杠杆: {leverage}x | 止损: {stoploss_pct}%")
            logger.info(f"  支撑: {key_support} | 阻力: {key_resistance} | RSI: {rsi_value} | 趋势强度: {trend_strength}")
            logger.info(f"  理由: {reason}")

            return {
                "success": True,
                "message": f"做多信号已发出 - 置信度{confidence_score}，挂单价{limit_price}，杠杆{leverage}x",
                "pair": pair,
                "limit_price": limit_price,
                "leverage": leverage,
                "stoploss_pct": stoploss_pct,
                "confidence_score": confidence_score,
                "key_support": key_support,
                "key_resistance": key_resistance,
                "rsi_value": rsi_value,
                "trend_strength": trend_strength,
                "reason": reason
            }

        except Exception as e:
            logger.error(f"发出做多信号失败: {e}")
            return {"success": False, "message": str(e)}

    def signal_entry_short(
        self,
        pair: str,
        limit_price: float,
        leverage: float,
        stoploss_pct: float,
        confidence_score: float,
        key_support: float,
        key_resistance: float,
        rsi_value: float,
        trend_strength: str,
        reason: str
    ) -> Dict[str, Any]:
        """
        发出做空信号

        Args:
            pair: 交易对
            limit_price: 挂单价格
            leverage: 杠杆倍数
            stoploss_pct: 止损百分比（负数）
            confidence_score: 决策置信度
            key_support: 关键支撑位
            key_resistance: 关键阻力位
            rsi_value: RSI数值
            trend_strength: 趋势强度
            reason: 开仓理由

        Returns:
            执行结果
        """
        try:
            # 验证参数
            if leverage < 1 or leverage > 100:
                return {"success": False, "message": "杠杆必须在1-100之间"}

            if stoploss_pct >= 0:
                return {"success": False, "message": "止损必须为负数"}

            if limit_price <= 0:
                return {"success": False, "message": "挂单价格必须大于0"}

            if confidence_score < 1 or confidence_score > 100:
                return {"success": False, "message": "置信度必须在1-100之间"}

            # 缓存信号
            self._signal_cache[pair] = {
                "action": "enter_short",
                "limit_price": limit_price,
                "leverage": leverage,
                "stoploss_pct": stoploss_pct,
                "confidence_score": confidence_score,
                "key_support": key_support,
                "key_resistance": key_resistance,
                "rsi_value": rsi_value,
                "trend_strength": trend_strength,
                "reason": reason
            }

            # 同时设置杠杆和止损
            if not hasattr(self.strategy, '_leverage_cache'):
                self.strategy._leverage_cache = {}
            if not hasattr(self.strategy, '_stoploss_cache'):
                self.strategy._stoploss_cache = {}

            self.strategy._leverage_cache[pair] = leverage
            self.strategy._stoploss_cache[pair] = stoploss_pct

            logger.info(f"[做空信号] {pair} | 置信度: {confidence_score} | 挂单价: {limit_price} | 杠杆: {leverage}x | 止损: {stoploss_pct}%")
            logger.info(f"  支撑: {key_support} | 阻力: {key_resistance} | RSI: {rsi_value} | 趋势强度: {trend_strength}")
            logger.info(f"  理由: {reason}")

            return {
                "success": True,
                "message": f"做空信号已发出 - 置信度{confidence_score}，挂单价{limit_price}，杠杆{leverage}x",
                "pair": pair,
                "limit_price": limit_price,
                "leverage": leverage,
                "stoploss_pct": stoploss_pct,
                "confidence_score": confidence_score,
                "key_support": key_support,
                "key_resistance": key_resistance,
                "rsi_value": rsi_value,
                "trend_strength": trend_strength,
                "reason": reason
            }

        except Exception as e:
            logger.error(f"发出做空信号失败: {e}")
            return {"success": False, "message": str(e)}

    def signal_exit(
        self,
        pair: str,
        limit_price: float,
        confidence_score: float,
        rsi_value: float,
        reason: str
    ) -> Dict[str, Any]:
        """
        发出平仓信号

        Args:
            pair: 交易对
            limit_price: 挂单价格
            confidence_score: 决策置信度
            rsi_value: RSI数值
            reason: 平仓理由

        Returns:
            执行结果
        """
        try:
            if limit_price <= 0:
                return {"success": False, "message": "挂单价格必须大于0"}

            if confidence_score < 1 or confidence_score > 100:
                return {"success": False, "message": "置信度必须在1-100之间"}

            self._signal_cache[pair] = {
                "action": "exit",
                "limit_price": limit_price,
                "confidence_score": confidence_score,
                "rsi_value": rsi_value,
                "reason": reason
            }

            logger.info(f"[平仓信号] {pair} | 置信度: {confidence_score} | 挂单价: {limit_price} | RSI: {rsi_value}")
            logger.info(f"  理由: {reason}")

            return {
                "success": True,
                "message": f"平仓信号已发出 - 置信度{confidence_score}，挂单价{limit_price}",
                "pair": pair,
                "limit_price": limit_price,
                "confidence_score": confidence_score,
                "rsi_value": rsi_value,
                "reason": reason
            }

        except Exception as e:
            logger.error(f"发出平仓信号失败: {e}")
            return {"success": False, "message": str(e)}

    def adjust_position(
        self,
        pair: str,
        adjustment_pct: float,
        limit_price: float,
        confidence_score: float,
        key_support: float,
        key_resistance: float,
        reason: str
    ) -> Dict[str, Any]:
        """
        调整仓位（加仓/减仓）

        Args:
            pair: 交易对
            adjustment_pct: 调整百分比 (正数加仓，负数减仓)
            limit_price: 挂单价格
            confidence_score: 决策置信度
            key_support: 关键支撑位
            key_resistance: 关键阻力位
            reason: 调整理由

        Returns:
            执行结果
        """
        try:
            if limit_price <= 0:
                return {"success": False, "message": "挂单价格必须大于0"}

            if adjustment_pct == 0:
                return {"success": False, "message": "调整幅度不能为0"}

            if confidence_score < 1 or confidence_score > 100:
                return {"success": False, "message": "置信度必须在1-100之间"}

            # 缓存调整信号
            if not hasattr(self.strategy, '_position_adjustment_cache'):
                self.strategy._position_adjustment_cache = {}

            self.strategy._position_adjustment_cache[pair] = {
                "adjustment_pct": adjustment_pct,
                "limit_price": limit_price,
                "confidence_score": confidence_score,
                "key_support": key_support,
                "key_resistance": key_resistance,
                "reason": reason
            }

            action = "加仓" if adjustment_pct > 0 else "减仓"
            logger.info(f"[{action}] {pair} | 置信度: {confidence_score} | 幅度: {abs(adjustment_pct):.1f}% | 挂单价: {limit_price}")
            logger.info(f"  支撑: {key_support} | 阻力: {key_resistance}")
            logger.info(f"  理由: {reason}")

            return {
                "success": True,
                "message": f"{action} {abs(adjustment_pct):.1f}% - 置信度{confidence_score}",
                "pair": pair,
                "adjustment_pct": adjustment_pct,
                "limit_price": limit_price,
                "confidence_score": confidence_score,
                "key_support": key_support,
                "key_resistance": key_resistance,
                "reason": reason
            }

        except Exception as e:
            logger.error(f"调整仓位失败: {e}")
            return {"success": False, "message": str(e)}

    def signal_hold(
        self,
        pair: str,
        confidence_score: float,
        rsi_value: float,
        reason: str
    ) -> Dict[str, Any]:
        """
        保持持仓不动

        Args:
            pair: 交易对
            confidence_score: 决策置信度
            rsi_value: RSI数值
            reason: 保持理由

        Returns:
            执行结果
        """
        try:
            if confidence_score < 1 or confidence_score > 100:
                return {"success": False, "message": "置信度必须在1-100之间"}

            self._signal_cache[pair] = {
                "action": "hold",
                "confidence_score": confidence_score,
                "rsi_value": rsi_value,
                "reason": reason
            }

            logger.debug(f"[保持] {pair} | 置信度: {confidence_score} | RSI: {rsi_value}")
            logger.debug(f"  理由: {reason}")

            return {
                "success": True,
                "message": f"持仓保持不变 - 置信度{confidence_score}",
                "pair": pair,
                "confidence_score": confidence_score,
                "rsi_value": rsi_value,
                "reason": reason
            }

        except Exception as e:
            logger.error(f"保持信号失败: {e}")
            return {"success": False, "message": str(e)}

    def signal_wait(
        self,
        pair: str,
        confidence_score: float,
        rsi_value: float,
        reason: str
    ) -> Dict[str, Any]:
        """
        空仓等待观望

        Args:
            pair: 交易对
            confidence_score: 决策置信度
            rsi_value: RSI数值
            reason: 等待理由

        Returns:
            执行结果
        """
        try:
            if confidence_score < 1 or confidence_score > 100:
                return {"success": False, "message": "置信度必须在1-100之间"}

            self._signal_cache[pair] = {
                "action": "wait",
                "confidence_score": confidence_score,
                "rsi_value": rsi_value,
                "reason": reason
            }

            logger.debug(f"[等待] {pair} | 置信度: {confidence_score} | RSI: {rsi_value}")
            logger.debug(f"  理由: {reason}")

            return {
                "success": True,
                "message": f"空仓等待 - 置信度{confidence_score}",
                "pair": pair,
                "confidence_score": confidence_score,
                "rsi_value": rsi_value,
                "reason": reason
            }

        except Exception as e:
            logger.error(f"等待信号失败: {e}")
            return {"success": False, "message": str(e)}

    def get_signal(self, pair: str) -> Optional[Dict[str, Any]]:
        """获取缓存的信号"""
        return self._signal_cache.get(pair)

    def clear_signals(self):
        """清空信号缓存"""
        self._signal_cache.clear()
