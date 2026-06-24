"""LLM 客户端封装，支持 Anthropic 和 OpenAI 兼容接口。"""

import json
import os
import time
from dataclasses import dataclass
from typing import Iterator


@dataclass
class ToolCall:
    """LLM 发起的工具调用。"""

    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    """LLM 返回的结构化响应。"""

    text: str | None = None
    tool_calls: list[ToolCall] | None = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


class LLMClient:
    """统一的 LLM 调用入口。"""

    def __init__(self, provider: str | None = None, model: str | None = None, api_key: str | None = None):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "anthropic")).lower()
        self.model = model or os.getenv("DEFAULT_MODEL", "claude-3-5-sonnet-20241022")

        if self.provider == "anthropic":
            import anthropic

            kwargs = {"api_key": api_key or os.getenv("ANTHROPIC_API_KEY")}
            base_url = os.getenv("ANTHROPIC_BASE_URL")
            if base_url:
                kwargs["base_url"] = base_url
            self._client = anthropic.Anthropic(**kwargs)
            self._async_client = anthropic.AsyncAnthropic(**kwargs)
        elif self.provider == "openai":
            import openai

            kwargs = {"api_key": api_key or os.getenv("OPENAI_API_KEY")}
            base_url = os.getenv("OPENAI_BASE_URL")
            if base_url:
                kwargs["base_url"] = base_url
            self._client = openai.OpenAI(**kwargs)
            self._async_client = openai.AsyncOpenAI(**kwargs)
        else:
            raise ValueError(f"不支持的 LLM provider: {self.provider}")

    def chat(self, system_prompt: str, messages: list[dict], tools: list[dict] | None = None, stream: bool = False) -> str | Iterator[str]:
        """同步聊天调用，带自动重试。

        Args:
            system_prompt: 系统提示词
            messages: 消息列表，格式 [{"role": "user"/"assistant", "content": "..."}]
            tools: 工具定义列表（OpenAI 格式）
            stream: 是否流式返回

        Returns:
            完整回复字符串，或流式片段迭代器
        """
        if self.provider == "anthropic":
            return self._anthropic_chat(system_prompt, messages, tools, stream)
        return self._openai_chat(system_prompt, messages, tools, stream)

    def _anthropic_chat(self, system_prompt: str, messages: list[dict], tools: list[dict] | None, stream: bool):
        kwargs = {
            "model": self.model,
            "max_tokens": 4096,
            "system": system_prompt,
            "messages": messages,
        }
        if tools:
            # Anthropic tools 格式与 OpenAI 略有不同，简单转换
            anthropic_tools = []
            for t in tools:
                anthropic_tools.append({
                    "name": t["function"]["name"],
                    "description": t["function"]["description"],
                    "input_schema": t["function"]["parameters"],
                })
            kwargs["tools"] = anthropic_tools

        # 重试机制：Kimi API 偶尔返回空 content，最多重试 2 次
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                if stream:
                    def _stream():
                        with self._client.messages.stream(**kwargs) as stream_obj:
                            for text in stream_obj.text_stream:
                                yield text
                    return _stream()

                response = self._client.messages.create(**kwargs)
                # 防御性检查：部分模型/API 偶尔返回 content 为空
                if not response.content:
                    if attempt < max_retries:
                        time.sleep(0.5 * (attempt + 1))
                        continue
                    return "(模型未返回有效内容，请重试)"
                return response.content[0].text
            except Exception as e:
                if attempt < max_retries:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise RuntimeError(f"Anthropic/Kimi API 调用失败 [{type(e).__name__}]: {e}") from e

    def _openai_chat(self, system_prompt: str, messages: list[dict], tools: list[dict] | None, stream: bool):
        msgs = [{"role": "system", "content": system_prompt}] + messages
        kwargs = {
            "model": self.model,
            "messages": msgs,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        try:
            if stream:
                def _stream():
                    for chunk in self._client.chat.completions.create(**kwargs, stream=True):
                        delta = chunk.choices[0].delta.content
                        if delta:
                            yield delta
                return _stream()

            response = self._client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"OpenAI API 调用失败 [{type(e).__name__}]: {e}") from e

    # ── 原生 Function Calling ──────────────────────────────

    def chat_with_tools(self, system_prompt: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        """支持原生工具调用的聊天接口，返回结构化响应。

        Args:
            system_prompt: 系统提示词
            messages: 消息列表（支持字符串 content 或 block 列表）
            tools: 工具定义列表（OpenAI 格式，内部会自动转换）

        Returns:
            LLMResponse，包含自然语言文本和/或工具调用列表
        """
        if self.provider == "anthropic":
            return self._anthropic_chat_with_tools(system_prompt, messages, tools)
        return self._openai_chat_with_tools(system_prompt, messages, tools)

    def _anthropic_chat_with_tools(self, system_prompt: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        """Anthropic 原生工具调用实现。"""
        # 转换工具格式
        anthropic_tools = []
        for t in tools:
            anthropic_tools.append({
                "name": t["function"]["name"],
                "description": t["function"]["description"],
                "input_schema": t["function"]["parameters"],
            })

        kwargs = {
            "model": self.model,
            "max_tokens": 4096,
            "system": system_prompt,
            "messages": messages,
            "tools": anthropic_tools,
        }

        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                response = self._client.messages.create(**kwargs)

                if not response.content:
                    if attempt < max_retries:
                        time.sleep(0.5 * (attempt + 1))
                        continue
                    return LLMResponse(text="(模型未返回有效内容，请重试)")

                text_parts = []
                tool_calls = []

                for block in response.content:
                    if block.type == "text":
                        text_parts.append(block.text)
                    elif block.type == "tool_use":
                        tool_calls.append(ToolCall(
                            id=block.id,
                            name=block.name,
                            arguments=block.input,
                        ))

                return LLMResponse(
                    text="".join(text_parts).strip() if text_parts else None,
                    tool_calls=tool_calls if tool_calls else None,
                )

            except Exception as e:
                if attempt < max_retries:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise RuntimeError(f"Anthropic/Kimi API 调用失败 [{type(e).__name__}]: {e}") from e

    def _openai_chat_with_tools(self, system_prompt: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        """OpenAI 原生工具调用实现。"""
        msgs = [{"role": "system", "content": system_prompt}] + messages
        kwargs = {
            "model": self.model,
            "messages": msgs,
            "tools": tools,
            "tool_choice": "auto",
        }

        try:
            response = self._client.chat.completions.create(**kwargs)
            msg = response.choices[0].message

            if msg.tool_calls:
                tool_calls = []
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}
                    tool_calls.append(ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=args,
                    ))
                return LLMResponse(text=msg.content, tool_calls=tool_calls)

            return LLMResponse(text=msg.content)

        except Exception as e:
            raise RuntimeError(f"OpenAI API 调用失败 [{type(e).__name__}]: {e}") from e
