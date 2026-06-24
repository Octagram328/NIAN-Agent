"""中期记忆：本次会话的阶段性摘要。"""

import json
from pathlib import Path
from src.llm.client import LLMClient


SUMMARY_PROMPT = """请总结以下对话的核心进展。要求：
1. 保留用户的核心意图和关键需求
2. 记录已完成的步骤和当前状态
3. 指出下一步可能需要做什么
4. 用 2-3 句话简洁表达

对话历史：
{history}

摘要："""


class MidTermMemory:
    """每 N 轮对话自动生成一次摘要，作为本次会话的"中期记忆"。

    中期记忆比短期记忆持久（跨轮保留），比长期记忆易变（仅当前会话有效）。
    """

    def __init__(self, summary_interval: int = 6, data_dir: str | None = None):
        """Args:
            summary_interval: 每多少条消息触发一次摘要更新（默认 6 条）
        """
        self.summary_interval = summary_interval
        self.summary: str = ""
        self._message_count_since_update = 0
        self._data_dir = Path(data_dir or "data")
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._summary_file = self._data_dir / "mid_term_summary.json"
        self._load()

    def _load(self) -> None:
        """从文件加载上次的中期记忆。"""
        if self._summary_file.exists():
            try:
                with open(self._summary_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.summary = data.get("summary", "")
            except (json.JSONDecodeError, OSError):
                pass

    def save(self) -> None:
        """保存当前中期记忆到文件。"""
        with open(self._summary_file, "w", encoding="utf-8") as f:
            json.dump({"summary": self.summary}, f, ensure_ascii=False, indent=2)

    def clear(self) -> None:
        """清空中期记忆（新会话时调用）。"""
        self.summary = ""
        self._message_count_since_update = 0
        if self._summary_file.exists():
            self._summary_file.unlink()

    def maybe_update(self, messages: list[dict], llm_client: LLMClient | None = None) -> bool:
        """检查是否需要更新摘要，若需要则调用 LLM 生成。

        Returns:
            是否发生了更新
        """
        self._message_count_since_update += 1
        if self._message_count_since_update < self.summary_interval:
            return False

        if not llm_client:
            return False

        # 构造对话历史文本
        history_lines = []
        for msg in messages:
            role_label = "用户" if msg["role"] == "user" else "助手"
            history_lines.append(f"{role_label}: {msg['content']}")
        history_text = "\n".join(history_lines[-20:])  # 只取最近 20 条

        prompt = SUMMARY_PROMPT.format(history=history_text)
        try:
            new_summary = llm_client.chat(
                system_prompt="你是一个对话摘要助手，擅长提炼关键信息。",
                messages=[{"role": "user", "content": prompt}],
            )
            if isinstance(new_summary, str) and new_summary.strip():
                self.summary = new_summary.strip()
                self._message_count_since_update = 0
                self.save()
                return True
        except Exception:
            pass
        return False

    def get_system_message(self) -> dict | None:
        """将中期记忆包装为系统消息注入上下文。"""
        if not self.summary:
            return None
        return {
            "role": "system",
            "content": self.summary,
        }
