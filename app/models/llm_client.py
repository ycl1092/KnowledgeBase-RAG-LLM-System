"""
LLM 客户端

支持 DeepSeek 和 OpenAI 兼容接口，带流式输出和自动重试。
"""

import time
from typing import Optional

from openai import OpenAI, Stream
from openai.types.chat import ChatCompletionChunk

from app.core.config import settings
from app.core.logger import logger


class LLMClient:
    """统一 LLM 调用客户端"""

    def __init__(self):
        self.api_key = settings.LLM_API_KEY
        self.base_url = settings.LLM_BASE_URL

        if not self.api_key:
            logger.warning("LLM_API_KEY 未配置，将使用 Mock 模式")
            self.client = None
        else:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        self.model = settings.LLM_MODEL
        self.temperature = settings.LLM_TEMPERATURE
        self.max_tokens = settings.LLM_MAX_TOKENS

    def chat(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """非流式调用，返回完整文本"""
        if self.client is None:
            return self._mock_reply(messages)

        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature or self.temperature,
                    max_tokens=max_tokens or self.max_tokens,
                    stream=False,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                logger.warning(f"LLM 调用失败 (第{attempt+1}次): {e}")
                if attempt < 2:
                    time.sleep(1 * (attempt + 1))
                else:
                    logger.error(f"LLM 调用重试耗尽，降级到 Mock: {e}")
                    return self._mock_reply(messages)

    def chat_stream(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        """流式调用，逐 chunk 返回"""
        if self.client is None:
            yield self._mock_reply(messages)
            return

        for attempt in range(3):
            try:
                stream: Stream[ChatCompletionChunk] = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature or self.temperature,
                    max_tokens=max_tokens or self.max_tokens,
                    stream=True,
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        yield delta.content
                return
            except Exception as e:
                logger.warning(f"LLM 流式调用失败 (第{attempt+1}次): {e}")
                if attempt < 2:
                    time.sleep(1 * (attempt + 1))
                else:
                    logger.error(f"LLM 流式调用重试耗尽: {e}")
                    yield self._mock_reply(messages)
                    return

    def _mock_reply(self, messages: list[dict]) -> str:
        last = messages[-1]["content"] if messages else ""
        return f"（Mock）已收到您的提问，配置 LLM_API_KEY 后可获得真实回答。您的问题是：{last[:50]}"

    @property
    def is_ready(self) -> bool:
        return self.client is not None


llm = LLMClient()
