"""
LLM客户端模块
负责与LLM API交互，支持函数调用
"""
import logging
from typing import Dict, Any, List, Optional
import requests
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class LLMClient:
    """LLM API客户端"""

    def __init__(self, llm_config: Dict[str, Any], function_executor):
        """
        初始化LLM客户端

        Args:
            llm_config: LLM配置
            function_executor: 函数执行器
        """
        self.api_base = llm_config.get("api_base", "http://host.docker.internal:3120")
        self.api_key = llm_config.get("api_key", "")
        self.model = llm_config.get("model", "qwen/qwen3-coder-30b")
        self.temperature = None  # 使用 LM Studio 的配置
        self.max_tokens = None  # 使用 LM Studio 的配置
        self.timeout = None  # 不设置超时限制，允许思考模型有足够时间推理

        self.function_executor = function_executor

        # 对话历史(用于上下文管理)
        self.conversation_history: List[Dict[str, Any]] = []
        self.max_history_length = 10  # 保留最近N轮对话

        logger.info(f"LLM客户端已初始化: {self.model}")

    def call_with_functions(
        self,
        messages: List[Dict[str, str]],
        functions: Optional[List[Dict[str, Any]]] = None,
        max_iterations: int = 5
    ) -> Dict[str, Any]:
        """
        调用LLM并支持函数调用

        Args:
            messages: 消息列表
            functions: 可用的函数列表
            max_iterations: 最大迭代次数(防止无限循环)

        Returns:
            LLM响应和执行结果
        """
        if functions is None:
            functions = self.function_executor.get_all_tools_schema()

        iteration = 0
        current_messages = messages.copy()
        function_call_history = []

        while iteration < max_iterations:
            iteration += 1

            logger.debug(f"🔄 迭代 {iteration}/{max_iterations} 开始")

            try:
                # 调用LLM API
                response = self._call_api(current_messages, functions)

                if not response:
                    return {
                        "success": False,
                        "error": "API调用失败",
                        "iteration": iteration
                    }

                # 解析响应
                choice = response.get("choices", [{}])[0]
                message = choice.get("message", {})
                finish_reason = choice.get("finish_reason", "")

                # 提取消息内容（兼容推理模型的特殊格式）
                message_content = self._extract_message_content(message)

                # 检查是否有函数调用
                tool_calls = message.get("tool_calls", [])

                if not tool_calls or finish_reason == "stop":
                    # 没有函数调用或已完成，返回最终响应
                    logger.debug(f"✅ 决策完成 (迭代{iteration}, 原因: {finish_reason or '无函数调用'})")
                    return {
                        "success": True,
                        "message": message_content,
                        "function_calls": function_call_history,
                        "iterations": iteration,
                        "finish_reason": finish_reason
                    }

                # 执行函数调用
                logger.debug(f"📞 本次迭代需要调用 {len(tool_calls)} 个函数")
                function_results = []
                should_terminate = False  # 是否遇到终止性函数

                for tool_call in tool_calls:
                    func_name = tool_call.get("function", {}).get("name", "")
                    func_args_str = tool_call.get("function", {}).get("arguments", "{}")

                    try:
                        func_args = json.loads(func_args_str) if isinstance(func_args_str, str) else func_args_str
                    except json.JSONDecodeError as e:
                        logger.error(f"解析函数参数失败: {e}")
                        func_args = {}

                    # 执行函数
                    result = self.function_executor.execute_function(func_name, func_args)

                    # 检查是否为终止性函数
                    if result.get("_is_terminal", False):
                        should_terminate = True
                        logger.info(f"🛑 检测到终止性函数 '{func_name}'，决策流程将结束")

                    # 记录
                    function_call_history.append({
                        "function": func_name,
                        "arguments": func_args,
                        "result": result
                    })

                    function_results.append({
                        "role": "tool",
                        "tool_call_id": tool_call.get("id", ""),
                        "name": func_name,
                        "content": json.dumps(result, ensure_ascii=False)
                    })

                # 如果遇到终止性函数，立即返回
                if should_terminate:
                    logger.info(f"✅ 决策完成 (迭代{iteration}, 调用终止性函数)")
                    return {
                        "success": True,
                        "message": message_content,
                        "function_calls": function_call_history,
                        "iterations": iteration,
                        "finish_reason": "terminal_function"
                    }

                # 将函数调用结果添加到消息历史
                current_messages.append(message)
                current_messages.extend(function_results)

            except Exception as e:
                logger.error(f"LLM调用失败 (迭代{iteration}): {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "iteration": iteration,
                    "function_calls": function_call_history
                }

        # 达到最大迭代次数（极少发生）
        logger.debug(f"达到最大迭代次数: {max_iterations}")
        return {
            "success": False,
            "error": "达到最大迭代次数",
            "iterations": max_iterations,
            "function_calls": function_call_history
        }

    def _call_api(
        self,
        messages: List[Dict[str, str]],
        functions: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        调用LLM API

        Args:
            messages: 消息列表
            functions: 函数列表

        Returns:
            API响应
        """
        try:
            url = f"{self.api_base}/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            payload = {
                "model": self.model,
                "messages": messages,
                "tools": [{"type": "function", "function": f} for f in functions],
                "tool_choice": "auto"
            }

            # 只添加非 None 的可选参数
            if self.temperature is not None:
                payload["temperature"] = self.temperature
            if self.max_tokens is not None:
                payload["max_tokens"] = self.max_tokens

            logger.debug(f"调用LLM API: {self.model}")

            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=self.timeout
            )

            if response.status_code != 200:
                logger.error(f"API返回错误: {response.status_code} - {response.text}")
                return None

            return response.json()

        except requests.Timeout:
            logger.error("API请求超时")
            return None
        except Exception as e:
            logger.error(f"API调用异常: {e}")
            return None

    def _extract_message_content(self, message: Dict[str, Any]) -> str:
        """
        提取消息内容，兼容推理模型的特殊响应格式

        Args:
            message: API返回的message对象

        Returns:
            提取的消息内容
        """
        # 检查并记录思考过程（如果有）
        think = message.get("think", "")
        reasoning = message.get("reasoning", "")
        reasoning_content = message.get("reasoning_content", "")

        if think:
            logger.info(f"[模型思考过程]\n{think}")
        elif reasoning:
            logger.info(f"[模型推理过程]\n{reasoning}")
        elif reasoning_content:
            logger.info(f"[模型推理内容]\n{reasoning_content}")

        # 优先使用 content 字段（标准格式 - 最终决策）
        content = message.get("content", "")
        if content:
            return content

        # 如果没有 content，尝试使用 reasoning 相关字段
        # 某些推理模型可能只返回 reasoning 而不返回 content
        if reasoning:
            logger.info("未找到 content 字段，使用 reasoning 作为响应")
            return reasoning

        if think:
            logger.info("未找到 content 字段，使用 think 作为响应")
            return think

        if reasoning_content:
            logger.info("未找到 content 字段，使用 reasoning_content 作为响应")
            return reasoning_content

        # 如果有tool_calls，说明LLM直接调用函数而没有输出文本，这是正常的
        if message.get("tool_calls"):
            logger.debug("消息中只有tool_calls，无文本内容（正常）")
            return ""

        # 如果既没有内容也没有tool_calls，才是异常情况
        logger.warning("消息中未找到 content、think、reasoning、reasoning_content 或 tool_calls 字段")
        return ""

    def simple_call(
        self,
        messages: List[Dict[str, str]]
    ) -> Optional[str]:
        """
        简单调用(不使用函数调用)

        Args:
            消息列表

        Returns:
            LLM响应文本
        """
        try:
            url = f"{self.api_base}/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            payload = {
                "model": self.model,
                "messages": messages
            }

            # 只添加非 None 的可选参数
            if self.temperature is not None:
                payload["temperature"] = self.temperature
            if self.max_tokens is not None:
                payload["max_tokens"] = self.max_tokens

            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=self.timeout
            )

            if response.status_code != 200:
                logger.error(f"API返回错误: {response.status_code}")
                return None

            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

            return content

        except Exception as e:
            logger.error(f"简单调用失败: {e}")
            return None

    def manage_context_window(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 6000
    ) -> List[Dict[str, str]]:
        """
        管理上下文窗口，避免超出token限制

        Args:
            messages: 消息列表
            max_tokens: 最大token数

        Returns:
            精简后的消息列表
        """
        # 简单估算: 1 token ≈ 2.5 字符
        estimated_tokens = sum(len(m.get("content", "")) for m in messages) / 2.5

        if estimated_tokens <= max_tokens:
            return messages

        # 保留系统消息和最近的用户消息
        system_messages = [m for m in messages if m.get("role") == "system"]
        other_messages = [m for m in messages if m.get("role") != "system"]

        # 从后往前保留消息，直到接近token限制
        kept_messages = []
        current_tokens = sum(len(m.get("content", "")) for m in system_messages) / 2.5

        for msg in reversed(other_messages):
            msg_tokens = len(msg.get("content", "")) / 2.5
            if current_tokens + msg_tokens > max_tokens * 0.9:  # 保留10%余量
                break
            kept_messages.insert(0, msg)
            current_tokens += msg_tokens

        return system_messages + kept_messages

    def add_to_history(self, role: str, content: str):
        """添加消息到历史"""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

        # 限制历史长度
        if len(self.conversation_history) > self.max_history_length * 2:
            self.conversation_history = self.conversation_history[-self.max_history_length * 2:]

    def clear_history(self):
        """清空对话历史"""
        self.conversation_history.clear()
        logger.info("对话历史已清空")

    def get_history(self, include_timestamp: bool = False) -> List[Dict[str, Any]]:
        """获取对话历史"""
        if include_timestamp:
            return self.conversation_history.copy()

        return [
            {"role": m["role"], "content": m["content"]}
            for m in self.conversation_history
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "model": self.model,
            "api_base": self.api_base,
            "conversation_length": len(self.conversation_history),
            "max_history_length": self.max_history_length
        }
