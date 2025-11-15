"""
市场情绪数据获取模块
获取免费的市场情绪指标：Fear & Greed Index, Funding Rate
"""

import requests
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class MarketSentiment:
    """市场情绪数据获取器"""

    def __init__(self):
        self.fear_greed_cache = None
        self.fear_greed_cache_time = None
        self.long_short_cache = {}  # {pair: {data, time}}
        self.cache_duration = 300  # 5分钟缓存

    def get_fear_greed_index(self) -> Optional[Dict[str, Any]]:
        """
        获取加密货币恐惧与贪婪指数
        来源: Alternative.me (免费API)

        返回:
        {
            'value': 50,  # 0-100
            'classification': 'Neutral',  # Extreme Fear, Fear, Neutral, Greed, Extreme Greed
            'timestamp': 1234567890,
            'trend': 'rising'  # rising, falling, stable
        }
        """
        # 检查缓存
        if self.fear_greed_cache and self.fear_greed_cache_time:
            if datetime.now() - self.fear_greed_cache_time < timedelta(seconds=self.cache_duration):
                return self.fear_greed_cache

        try:
            # Alternative.me Fear & Greed Index API
            # 获取最近30天数据，让模型看到完整的情绪变化历史
            url = "https://api.alternative.me/fng/?limit=30"
            response = requests.get(url, timeout=5)
            response.raise_for_status()

            data = response.json()
            if 'data' not in data or len(data['data']) == 0:
                logger.warning("Fear & Greed Index API 返回空数据")
                return None

            # 当前值
            current = data['data'][0]
            value = int(current['value'])
            classification = current['value_classification']
            timestamp = int(current['timestamp'])

            # 计算短期趋势（最近2天）
            trend = 'stable'
            if len(data['data']) >= 2:
                previous = int(data['data'][1]['value'])
                diff = value - previous
                if diff > 5:
                    trend = 'rising'
                elif diff < -5:
                    trend = 'falling'

            # 构建完整历史（带时间戳和分类）
            history_with_time = []
            for d in data['data']:
                history_with_time.append({
                    'value': int(d['value']),
                    'classification': d['value_classification'],
                    'timestamp': int(d['timestamp']),
                    'date': datetime.fromtimestamp(int(d['timestamp'])).strftime('%Y-%m-%d')
                })

            # 分析情绪持续时间
            duration_days = 0
            history_values = [int(d['value']) for d in data['data']]

            # 判断当前情绪持续了多少天（同一区间：极度恐惧<25, 恐惧25-45, 中性45-55, 贪婪55-75, 极度贪婪>75）
            def get_zone(v):
                if v < 25: return 'extreme_fear'
                elif v < 45: return 'fear'
                elif v < 55: return 'neutral'
                elif v < 75: return 'greed'
                else: return 'extreme_greed'

            current_zone = get_zone(value)
            for v in history_values:
                if get_zone(v) == current_zone:
                    duration_days += 1
                else:
                    break

            # 计算30天变化趋势
            if len(history_values) >= 7:
                week_change = history_values[0] - history_values[6]
                week_trend = 'rising' if week_change > 10 else ('falling' if week_change < -10 else 'stable')
            else:
                week_trend = trend

            if len(history_values) >= 30:
                month_change = history_values[0] - history_values[-1]
                month_trend = 'rising' if month_change > 15 else ('falling' if month_change < -15 else 'stable')
            else:
                month_trend = week_trend

            result = {
                'value': value,
                'classification': classification,
                'timestamp': timestamp,
                'trend': trend,  # 短期趋势（1-2天）
                'week_trend': week_trend,  # 周趋势
                'month_trend': month_trend,  # 月趋势
                'duration_days': duration_days,  # 当前情绪持续天数
                'history_30d': history_with_time,  # 最近30天历史（带时间戳）
                'history_values': history_values,  # 仅数值（用于快速趋势判断）
                'interpretation': self._interpret_fear_greed(value, trend, duration_days)
            }

            # 更新缓存
            self.fear_greed_cache = result
            self.fear_greed_cache_time = datetime.now()

            return result

        except requests.exceptions.RequestException as e:
            logger.error(f"获取 Fear & Greed Index 失败: {e}")
            return None
        except Exception as e:
            logger.error(f"处理 Fear & Greed Index 数据失败: {e}")
            return None

    def get_funding_rate(self, exchange, pair: str) -> Optional[Dict[str, Any]]:
        """
        获取资金费率

        参数:
            exchange: Freqtrade exchange 对象
            pair: 交易对，如 'BTC/USDT:USDT'

        返回:
        {
            'rate': 0.0001,  # 当前资金费率
            'rate_pct': 0.01,  # 百分比形式
            'next_funding_time': 1234567890,
            'interpretation': '多头极度过热(0.010%，多头付费给空头)'
        }
        """
        try:
            # 使用 ccxt 的 fetch_funding_rate 方法
            if hasattr(exchange, 'fetch_funding_rate'):
                funding_info = exchange.fetch_funding_rate(pair)

                if not funding_info or 'fundingRate' not in funding_info:
                    logger.warning(f"无法获取 {pair} 的资金费率")
                    return None

                rate = float(funding_info['fundingRate'])
                rate_pct = rate * 100  # 转换为百分比

                next_funding_time = funding_info.get('fundingTimestamp', 0)

                result = {
                    'rate': rate,
                    'rate_pct': rate_pct,
                    'next_funding_time': next_funding_time,
                    'interpretation': self._interpret_funding_rate(rate_pct)
                }

                return result
            else:
                logger.warning(f"交易所不支持 fetch_funding_rate")
                return None

        except Exception as e:
            logger.error(f"获取资金费率失败 {pair}: {e}")
            return None

    def _interpret_fear_greed(self, value: int, trend: str, duration_days: int = 0) -> str:
        """解释恐惧与贪婪指数 - 仅提供客观描述，不做主观判断"""
        interpretations = []

        # 当前状态 - 只描述情绪，不提供交易建议
        if value <= 20:
            interpretations.append("极度恐惧")
        elif value <= 40:
            interpretations.append("恐惧")
        elif value <= 60:
            interpretations.append("中性")
        elif value <= 80:
            interpretations.append("贪婪")
        else:
            interpretations.append("极度贪婪")

        # 持续时间分析 - 客观描述
        if duration_days > 0:
            interpretations.append(f"已持续{duration_days}天")
            if duration_days >= 5:
                if value <= 40:
                    interpretations.append("长期处于恐惧状态")
                elif value >= 60:
                    interpretations.append("长期处于贪婪状态")

        # 短期趋势
        if trend == 'rising':
            interpretations.append("情绪回升中")
        elif trend == 'falling':
            interpretations.append("情绪下降中")

        return ", ".join(interpretations)

    def _interpret_funding_rate(self, rate_pct: float) -> str:
        """
        解释资金费率 - 仅提供客观描述，不做主观判断

        资金费率含义:
        - 正值: 多头付费给空头 (多头过热)
        - 负值: 空头付费给多头 (空头过热)
        - 一般范围: -0.1% 到 0.1%
        """
        if rate_pct > 0.1:
            return f"多头极度过热({rate_pct:.3f}%，多头付费给空头)"
        elif rate_pct > 0.05:
            return f"多头过热({rate_pct:.3f}%)"
        elif rate_pct > -0.05:
            return f"市场平衡({rate_pct:.3f}%)"
        elif rate_pct > -0.1:
            return f"空头过热({rate_pct:.3f}%)"
        else:
            return f"空头极度过热({rate_pct:.3f}%，空头付费给多头)"

    def get_long_short_ratio(self, pair: str) -> Optional[Dict[str, Any]]:
        """
        获取币安多空比历史数据（30天，1小时间隔）

        参数:
            pair: 交易对，如 'BTC/USDT:USDT'

        返回:
        {
            'current_ratio': 1.2,  # 当前多空比
            'trend': 'bullish',  # bullish/bearish/neutral
            'interpretation': '多头占优',
            'history_24h': [...],  # 最近24小时数据
            'history_7d': [...],  # 最近7天数据
            'history_30d': [...],  # 最近30天数据（最多720个点）
            'extreme_level': 'normal'  # extreme_long/extreme_short/normal
        }
        """
        # 检查缓存
        if pair in self.long_short_cache:
            cache_entry = self.long_short_cache[pair]
            if datetime.now() - cache_entry['time'] < timedelta(seconds=self.cache_duration):
                return cache_entry['data']

        try:
            # 转换交易对格式：BTC/USDT:USDT -> BTCUSDT
            symbol = pair.replace('/USDT:USDT', 'USDT').replace('/', '')

            # 币安API：全局多空账户比
            url = "https://fapi.binance.com/futures/data/globalLongShortAccountRatio"
            params = {
                'symbol': symbol,
                'period': '1h',  # 1小时间隔
                'limit': 500  # 币安API最大限制，约20天数据
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            if not data or len(data) == 0:
                logger.warning(f"币安多空比API返回空数据: {symbol}")
                return None

            # 解析数据：[{longShortRatio, longAccount, shortAccount, timestamp}, ...]
            # Binance API返回的数据是按时间正序的（旧->新），timestamp递增
            # 注意：longAccount和shortAccount是小数比例（如0.654代表65.4%），需要乘100
            history = []
            for item in data:  # 直接使用正序数据（旧->新）
                ratio = float(item.get('longShortRatio', 0))
                if ratio > 0:
                    history.append({
                        'ratio': ratio,
                        'long_pct': float(item.get('longAccount', 0)) * 100,  # 转换为百分比
                        'short_pct': float(item.get('shortAccount', 0)) * 100,  # 转换为百分比
                        'timestamp': int(item.get('timestamp', 0))
                    })

            if len(history) == 0:
                return None

            # 当前多空比
            current = history[-1]
            current_ratio = current['ratio']

            # 分析趋势（最近24小时）
            if len(history) >= 24:
                ratio_24h_ago = history[-24]['ratio']
                change_24h = current_ratio - ratio_24h_ago

                if change_24h > 0.1:
                    trend = 'bullish'  # 多头增强
                elif change_24h < -0.1:
                    trend = 'bearish'  # 空头增强
                else:
                    trend = 'neutral'
            else:
                trend = 'neutral'

            # 判断极端情绪
            if current_ratio > 2.0:
                extreme_level = 'extreme_long'  # 多头过热
            elif current_ratio < 0.5:
                extreme_level = 'extreme_short'  # 空头过热
            elif current_ratio > 1.5:
                extreme_level = 'high_long'
            elif current_ratio < 0.7:
                extreme_level = 'high_short'
            else:
                extreme_level = 'normal'

            # 计算持续时间（当前情绪区间持续多久）
            def get_sentiment_zone(ratio):
                if ratio > 2.0: return 'extreme_long'
                elif ratio > 1.5: return 'high_long'
                elif ratio > 1.2: return 'slight_long'
                elif ratio > 0.8: return 'balanced'
                elif ratio > 0.5: return 'slight_short'
                else: return 'extreme_short'

            current_zone = get_sentiment_zone(current_ratio)
            duration_hours = 1  # 至少1小时
            for i in range(len(history) - 2, -1, -1):
                if get_sentiment_zone(history[i]['ratio']) == current_zone:
                    duration_hours += 1
                else:
                    break

            result = {
                'current_ratio': current_ratio,
                'long_pct': current['long_pct'],
                'short_pct': current['short_pct'],
                'trend': trend,
                'extreme_level': extreme_level,
                'duration_hours': duration_hours,
                'history_24h': [h['ratio'] for h in history[-24:]],
                'history_7d': [h['ratio'] for h in history[-168:]] if len(history) >= 168 else [h['ratio'] for h in history],
                'history_30d': history,  # 完整历史（包含ratio, long_pct, short_pct, timestamp）
                'interpretation': self._interpret_long_short_ratio(current_ratio, trend, extreme_level, duration_hours)
            }

            # 更新缓存
            self.long_short_cache[pair] = {
                'data': result,
                'time': datetime.now()
            }

            return result

        except Exception as e:
            logger.error(f"获取多空比失败 {pair}: {e}")
            return None

    def _interpret_long_short_ratio(self, ratio: float, trend: str, extreme: str, duration_hours: int) -> str:
        """解释多空比"""
        interpretations = []

        # 当前状态 - 只描述多空力量对比，不提供交易建议
        if ratio > 2.0:
            interpretations.append(f"多头极度过热({ratio:.2f})")
        elif ratio > 1.5:
            interpretations.append(f"多头偏强({ratio:.2f})")
        elif ratio > 1.2:
            interpretations.append(f"多头略占优({ratio:.2f})")
        elif ratio > 0.8:
            interpretations.append(f"多空平衡({ratio:.2f})")
        elif ratio > 0.5:
            interpretations.append(f"空头略占优({ratio:.2f})")
        else:
            interpretations.append(f"空头极度过热({ratio:.2f})")

        # 持续时间 - 客观描述
        if duration_hours >= 24:
            interpretations.append(f"已持续{duration_hours}小时")
            if extreme in ['extreme_long', 'extreme_short']:
                interpretations.append("长期处于极端状态")

        # 趋势
        if trend == 'bullish':
            interpretations.append("多头增强中")
        elif trend == 'bearish':
            interpretations.append("空头增强中")

        return ", ".join(interpretations)

    def get_combined_sentiment(self, exchange, pair: str) -> Dict[str, Any]:
        """
        获取综合市场情绪

        返回:
        {
            'fear_greed': {...},
            'funding_rate': {...},
            'overall_signal': 'bullish/bearish/neutral',
            'confidence': 'high/medium/low'
        }
        """
        fear_greed = self.get_fear_greed_index()
        funding_rate = self.get_funding_rate(exchange, pair)
        long_short = self.get_long_short_ratio(pair)

        # 综合判断 - 基于逆向思维的信号聚合
        # 注意：此信号仅供参考，实际决策需结合趋势、结构、持仓状态等完整上下文
        signals = []

        if fear_greed:
            fg_value = fear_greed['value']
            if fg_value <= 20:
                signals.append('bullish')  # 极度恐惧（逆向信号）
            elif fg_value >= 80:
                signals.append('bearish')  # 极度贪婪（逆向信号）

        if funding_rate:
            fr_pct = funding_rate['rate_pct']
            if fr_pct > 0.1:
                signals.append('bearish')  # 多头极度过热
            elif fr_pct < -0.1:
                signals.append('bullish')  # 空头极度过热

        if long_short:
            ls_ratio = long_short['current_ratio']
            if ls_ratio > 2.0:
                signals.append('bearish')  # 多头极度过热
            elif ls_ratio < 0.5:
                signals.append('bullish')  # 空头极度过热

        # 综合信号
        if len(signals) == 0:
            overall_signal = 'neutral'
            confidence = 'low'
        elif signals.count('bullish') > signals.count('bearish'):
            overall_signal = 'bullish'
            confidence = 'high' if len(signals) == 2 else 'medium'
        elif signals.count('bearish') > signals.count('bullish'):
            overall_signal = 'bearish'
            confidence = 'high' if len(signals) == 2 else 'medium'
        else:
            overall_signal = 'neutral'
            confidence = 'medium'

        return {
            'fear_greed': fear_greed,
            'funding_rate': funding_rate,
            'long_short': long_short,
            'overall_signal': overall_signal,
            'confidence': confidence
        }
