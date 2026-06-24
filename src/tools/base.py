"""工具基类定义。"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ToolResult:
    """工具执行结果。"""

    success: bool
    content: str
    error: str | None = None


class Tool(ABC):
    """所有工具的抽象基类。"""

    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称（英文，用于 LLM 识别）。"""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """工具描述（用于 LLM 判断何时调用）。"""
        ...

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """JSON Schema 格式的参数定义。"""
        ...

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """执行工具逻辑。"""
        ...

    def to_openai_schema(self) -> dict:
        """转换为 OpenAI / Anthropic 可用的工具定义格式。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
