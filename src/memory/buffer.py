"""短期记忆：对话上下文缓冲区。"""

from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class ConversationBuffer:
    """维护最近 N 轮对话的短期记忆。

    每轮 = user 消息 + assistant 消息（+ 可选的 tool 结果）。
    当轮数超过 max_turns 时，丢弃最旧的轮次。
    """

    max_turns: int = 10
    _messages: list[dict] = field(default_factory=list, repr=False)

    def add_message(self, role: str, content: str) -> None:
        """添加一条消息，自动截断超长内容以控制上下文大小。"""
        # 工具结果/助手回复超长时截断，保留首尾各一部分
        MAX_CONTENT = 600
        if len(content) > MAX_CONTENT:
            half = MAX_CONTENT // 2
            content = content[:half] + f"\n[…已截断，共 {len(content)} 字…]\n" + content[-half:]
        self._messages.append({"role": role, "content": content})
        self._enforce_limit()

    def add_messages(self, messages: list[dict]) -> None:
        """批量添加消息。"""
        for msg in messages:
            self._messages.append({"role": msg["role"], "content": msg["content"]})
        self._enforce_limit()

    def get_messages(self) -> list[dict]:
        """获取当前缓冲区的所有消息（副本）。"""
        return list(self._messages)

    def clear(self) -> None:
        """清空缓冲区。"""
        self._messages.clear()

    def _enforce_limit(self) -> None:
        """根据 max_turns 裁剪消息列表。

        这里简单按 'user' 消息数计算轮数，保留最近的完整轮次。
        """
        if self.max_turns <= 0:
            return

        user_indices = [i for i, m in enumerate(self._messages) if m["role"] == "user"]
        if len(user_indices) <= self.max_turns:
            return

        # 找到第 N 轮 user 消息的位置，保留从该位置开始的所有消息
        keep_from = user_indices[-self.max_turns]
        self._messages = self._messages[keep_from:]

    def __iter__(self) -> Iterator[dict]:
        yield from self._messages

    def __len__(self) -> int:
        return len(self._messages)
