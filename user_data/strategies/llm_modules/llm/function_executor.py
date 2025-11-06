"""
函数执行器模块
管理和执行LLM的函数调用
"""
import logging
from typing import Dict, Any, List, Callable, Optional
import json

logger = logging.getLogger(__name__)


class FunctionExecutor:
    """函数调用执行器"""

    # 终止性函数：调用后应该立即结束决策流程
    TERMINAL_FUNCTIONS = {
        "signal_entry_long",
        "signal_entry_short",
        "signal_exit",
        "signal_hold",
        "signal_wait",
        "adjust_position"
    }

    def __init__(self):
        """初始化函数执行器"""
        self.functions: Dict[str, Callable] = {}
        self.schemas: Dict[str, Dict[str, Any]] = {}

    def register_tool(
        self,
        name: str,
        func: Callable,
        schema: Dict[str, Any]
    ):
        """
        注册工具函数

        Args:
            name: 函数名
            func: 函数对象
            schema: OpenAI函数schema
        """
        self.functions[name] = func
        self.schemas[name] = schema
        logger.debug(f"已注册函数: {name}")

    def register_tools_from_instance(self, tool_instance, schemas: List[Dict[str, Any]]):
        """
        从工具实例批量注册函数

        Args:
            tool_instance: 工具实例(如TradingTools)
            schemas: 函数schema列表
        """
        for schema in schemas:
            func_name = schema["name"]
            if hasattr(tool_instance, func_name):
                func = getattr(tool_instance, func_name)
                self.register_tool(func_name, func, schema)
            else:
                logger.warning(f"工具实例缺少方法: {func_name}")

    def execute_function(
        self,
        name: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        执行函数调用

        Args:
            name: 函数名
            arguments: 函数参数

        Returns:
            执行结果
        """
        try:
            if name not in self.functions:
                error_msg = f"未知函数: {name}"
                logger.error(error_msg)
                return {
                    "success": False,
                    "error": error_msg,
                    "function": name
                }

            func = self.functions[name]

            # 记录函数调用
            logger.info(f"🔧 调用函数: {name}")
            logger.info(f"   参数: {json.dumps(arguments, ensure_ascii=False)}")

            # 执行函数
            result = func(**arguments)

            # 记录返回结果（简化版）
            if isinstance(result, dict):
                if result.get('success'):
                    logger.info(f"   ✅ 成功")
                else:
                    logger.warning(f"   ❌ 失败: {result.get('message', '未知错误')}")
            else:
                logger.info(f"   ✅ 完成")

            # 确保返回字典格式
            if not isinstance(result, dict):
                result = {"result": result}

            # 标记是否为终止性函数
            result["_is_terminal"] = self.is_terminal_function(name)

            return result

        except TypeError as e:
            error_msg = f"函数参数错误: {e}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "function": name,
                "arguments": arguments
            }

        except Exception as e:
            error_msg = f"函数执行失败: {e}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "function": name
            }

    def execute_function_calls(
        self,
        function_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        批量执行函数调用

        Args:
            function_calls: 函数调用列表

        Returns:
            执行结果列表
        """
        results = []

        for call in function_calls:
            name = call.get("name", "")
            arguments = call.get("arguments", {})

            # 如果arguments是字符串，尝试解析为JSON
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError as e:
                    logger.error(f"解析函数参数失败: {e}")
                    results.append({
                        "success": False,
                        "error": f"参数格式错误: {e}",
                        "function": name
                    })
                    continue

            result = self.execute_function(name, arguments)
            results.append(result)

        return results

    def get_all_tools_schema(self) -> List[Dict[str, Any]]:
        """
        获取所有工具的schema列表

        Returns:
            Schema列表
        """
        return list(self.schemas.values())

    def get_tools_by_category(self) -> Dict[str, List[str]]:
        """
        按类别获取工具

        Returns:
            分类的工具列表
        """
        categories = {
            "trading": [],
            "market_data": [],
            "risk_management": [],
            "rag": []
        }

        for func_name in self.functions.keys():
            if any(keyword in func_name for keyword in ["signal", "leverage", "stoploss", "lock", "adjust"]):
                categories["trading"].append(func_name)
            elif any(keyword in func_name for keyword in ["get_ohlcv", "get_technical", "get_orderbook", "get_funding", "get_market"]):
                categories["market_data"].append(func_name)
            elif any(keyword in func_name for keyword in ["balance", "position", "calculate", "check_risk"]):
                categories["risk_management"].append(func_name)
            elif any(keyword in func_name for keyword in ["query", "similar", "experience", "pattern", "lesson"]):
                categories["rag"].append(func_name)

        return categories

    def validate_function_call(
        self,
        name: str,
        arguments: Dict[str, Any]
    ) -> tuple[bool, Optional[str]]:
        """
        验证函数调用是否合法

        Args:
            name: 函数名
            arguments: 参数

        Returns:
            (是否合法, 错误信息)
        """
        # 检查函数是否存在
        if name not in self.functions:
            return False, f"未知函数: {name}"

        # 检查必需参数
        schema = self.schemas.get(name, {})
        required_params = schema.get("parameters", {}).get("required", [])

        for param in required_params:
            if param not in arguments:
                return False, f"缺少必需参数: {param}"

        return True, None

    def get_function_description(self, name: str) -> str:
        """获取函数描述"""
        schema = self.schemas.get(name, {})
        return schema.get("description", "无描述")

    def list_functions(self) -> List[str]:
        """列出所有已注册的函数"""
        return list(self.functions.keys())

    def is_terminal_function(self, name: str) -> bool:
        """
        检查是否为终止性函数

        Args:
            name: 函数名

        Returns:
            是否为终止性函数
        """
        return name in self.TERMINAL_FUNCTIONS

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        categories = self.get_tools_by_category()

        return {
            "total_functions": len(self.functions),
            "categories": {
                cat: len(funcs) for cat, funcs in categories.items()
            },
            "functions": self.list_functions()
        }
