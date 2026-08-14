"""
DeepSeek API 客户端
支持流式输出、上下文记忆、thinking 模式。
"""
import logging
from typing import AsyncIterator, Dict, List, Optional

import openai

from backend.config import CFG

logger = logging.getLogger(__name__)


class DeepSeekClient:
    """
    DeepSeek 聊天客户端。
    使用 OpenAI 兼容接口，支持 deepseek-v4-flash / deepseek-v4-pro。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ):
        self.api_key = api_key or CFG.DEEPSEEK_API_KEY
        self.base_url = base_url or CFG.DEEPSEEK_BASE_URL
        self.model = model or CFG.DEEPSEEK_MODEL
        self.reasoning_effort = reasoning_effort or CFG.DEEPSEEK_REASONING_EFFORT
        self.system_prompt = system_prompt or CFG.SYSTEM_PROMPT

        if not self.api_key:
            raise ValueError("DeepSeek API Key 未配置，请在 .env 中设置 DEEPSEEK_API_KEY")

        self.client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=60,
        )

        # 多轮对话历史
        self.history: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]
        self.max_history = 10

    def clear_history(self):
        """清空对话历史（保留系统提示词）。"""
        self.history = [{"role": "system", "content": self.system_prompt}]

    def _trim_history(self):
        """控制历史长度，保留最近 max_history 轮。"""
        if len(self.history) > self.max_history + 1:
            self.history = [self.history[0]] + self.history[-self.max_history :]

    async def chat(
        self,
        user_message: str,
        stream: bool = True,
        enable_thinking: bool = False,
    ) -> AsyncIterator[str]:
        """
        发送消息并流式返回 AI 回复文本。
        若 enable_thinking=True，则启用 reasoning_effort。
        """
        self.history.append({"role": "user", "content": user_message})
        self._trim_history()

        try:
            extra_body = {}
            if enable_thinking and self.reasoning_effort:
                extra_body["reasoning_effort"] = self.reasoning_effort

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=self.history,
                stream=stream,
                temperature=0.7,
                max_tokens=2048,
                extra_body=extra_body,
            )

            full_reply = ""
            if stream:
                async for chunk in response:
                    delta = chunk.choices[0].delta
                    # DeepSeek 的 reasoning 内容在 reasoning_content 字段
                    reasoning = getattr(delta, "reasoning_content", None)
                    content = delta.content or ""

                    if reasoning:
                        yield f"[thinking]{reasoning}[/thinking]"
                    if content:
                        full_reply += content
                        yield content
            else:
                content = response.choices[0].message.content or ""
                full_reply = content
                yield content

            # 保存助手回复到历史
            if full_reply:
                self.history.append({"role": "assistant", "content": full_reply})
        except Exception as e:
            logger.error("DeepSeek 调用失败: %s", e)
            error_msg = f"抱歉，与 DeepSeek 通信时出现异常：{str(e)}"
            yield error_msg

    async def chat_once(
        self,
        user_message: str,
        enable_thinking: bool = False,
    ) -> str:
        """非流式单次对话，返回完整文本。"""
        full = ""
        async for chunk in self.chat(user_message, stream=False, enable_thinking=enable_thinking):
            full += chunk
        return full
