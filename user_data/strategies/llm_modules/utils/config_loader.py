"""
配置加载器模块
负责加载和验证LLM策略的所有配置
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ConfigLoader:
    """配置加载和验证器"""

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置加载器

        Args:
            config_path: 配置文件路径，如果为None则使用默认路径
        """
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()
        self._validate_config()

    def _get_default_config_path(self) -> str:
        """获取默认配置文件路径"""
        # 从策略目录向上查找config.json
        current = Path(__file__).parent
        for _ in range(5):  # 最多向上查找5层
            config_file = current / "config.json"
            if config_file.exists():
                return str(config_file)
            current = current.parent

        # 如果找不到，返回默认路径
        return str(Path(__file__).parent.parent.parent.parent / "config.json")

    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"配置已从 {self.config_path} 加载")
            return config
        except FileNotFoundError:
            logger.warning(f"配置文件 {self.config_path} 不存在，使用默认配置")
            return self._get_default_config()
        except json.JSONDecodeError as e:
            logger.error(f"配置文件格式错误: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "llm_config": {
                "api_base": "http://localhost:11434/v1",
                "api_key": "sk-xxx",
                "model": "your-model-name",
                "temperature": 0.7,
                "max_tokens": 2000,
                "timeout": 60
            },
            "risk_management": {
                "max_leverage": 100,
                "max_position_pct": 50,
                "max_open_trades": 4,
                "allow_model_freedom": True,
                "emergency_stop_loss": -0.15
            },
            "experience_config": {
                "log_decisions": True,
                "log_trades": True,
                "decision_log_path": "./user_data/logs/llm_decisions.jsonl",
                "trade_log_path": "./user_data/logs/trade_experience.jsonl"
            },
            "context_config": {
                "max_context_tokens": 6000,
                "system_prompt_tokens": 500,
                "market_data_tokens": 800,
                "enable_context_compression": True,
                "include_timeframe_alignment": True,
                "include_timeframe_guidance": True,
                "include_multi_timeframe_data": True,
                "indicator_history_points": 20,
                "indicator_history_lookback": 100,
                "raw_kline_history_points": 0,
                "raw_kline_max_rows": 120,
                "raw_kline_stride": 1,
                "raw_kline_compact_format": True,
                "raw_kline_extra_fields": [
                    "ema_20",
                    "ema_50",
                    "rsi",
                    "macd",
                    "macd_signal",
                    "macd_hist"
                ],
                "multi_timeframe_compact_format": True,
                "multi_timeframe_max_rows": 120,
                "default_multi_timeframe_fields": [
                    "ema_20",
                    "ema_50",
                    "rsi",
                    "macd",
                    "macd_signal",
                    "macd_hist",
                    "atr",
                    "adx"
                ],
                "multi_timeframe_history": {}
            }
        }

    def _validate_config(self):
        """验证配置的有效性"""
        # 验证LLM配置
        llm_config = self.config.get("llm_config", {})
        required_llm_keys = ["api_base", "model"]
        for key in required_llm_keys:
            if key not in llm_config:
                logger.warning(f"LLM配置缺少必需字段: {key}")

        # 验证风险管理配置
        risk_config = self.config.get("risk_management", {})
        if risk_config.get("max_leverage", 0) > 100:
            logger.warning("最大杠杆超过100x，可能存在极高风险")

    def get_llm_config(self) -> Dict[str, Any]:
        """获取LLM配置"""
        return self.config.get("llm_config", {})

    def get_risk_config(self) -> Dict[str, Any]:
        """获取风险管理配置"""
        return self.config.get("risk_management", {})

    def get_experience_config(self) -> Dict[str, Any]:
        """获取经验系统配置"""
        return self.config.get("experience_config", {})

    def get_context_config(self) -> Dict[str, Any]:
        """获取上下文管理配置"""
        return self.config.get("context_config", {})

    def get_all_config(self) -> Dict[str, Any]:
        """获取完整配置"""
        return self.config

    def reload_config(self):
        """重新加载配置文件"""
        self.config = self._load_config()
        self._validate_config()
        logger.info("配置已重新加载")
