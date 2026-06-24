"""长期记忆：本地文件持久化存储（增强版，支持摘要与检索）。"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any


class FileMemoryStore:
    """将会话历史和用户偏好持久化到本地文件系统。

    存储结构：
        data/
        ├── preferences.json      # 用户偏好、常用设置
        ├── sessions/
        │   ├── 2026-06-23_001.json
        │   └── ...
        └── session_summaries.json  # 会话摘要索引（用于跨会话检索）
    """

    def __init__(self, data_dir: str | None = None):
        self.data_dir = Path(data_dir or "data")
        self.sessions_dir = self.data_dir / "sessions"
        self.preferences_file = self.data_dir / "preferences.json"
        self.summaries_file = self.data_dir / "session_summaries.json"

        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._summaries: list[dict] = self._load_summaries()

    # ---------- 用户偏好 ----------

    def load_preferences(self) -> dict[str, Any]:
        if not self.preferences_file.exists():
            return {}
        try:
            with open(self.preferences_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def save_preferences(self, prefs: dict[str, Any]) -> None:
        with open(self.preferences_file, "w", encoding="utf-8") as f:
            json.dump(prefs, f, ensure_ascii=False, indent=2)

    def update_preference(self, key: str, value: Any) -> None:
        prefs = self.load_preferences()
        prefs[key] = value
        self.save_preferences(prefs)

    # ---------- 会话历史 ----------

    def save_session(self, messages: list[dict], metadata: dict | None = None, summary: str = "") -> str:
        session_id = self._generate_session_id()
        session_data = {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "metadata": metadata or {},
            "summary": summary,
            "messages": messages,
        }
        filepath = self.sessions_dir / f"{session_id}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)

        # 同时更新摘要索引
        if summary:
            self._summaries.append({
                "session_id": session_id,
                "created_at": session_data["created_at"],
                "summary": summary,
                "keywords": self._extract_keywords(summary),
            })
            self._save_summaries()

        return str(filepath)

    def list_sessions(self, limit: int = 10) -> list[dict]:
        files = sorted(self.sessions_dir.glob("*.json"), key=os.path.getmtime, reverse=True)
        sessions = []
        for fp in files[:limit]:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                sessions.append({
                    "session_id": data.get("session_id"),
                    "created_at": data.get("created_at"),
                    "summary": data.get("summary", ""),
                    "metadata": data.get("metadata", {}),
                    "message_count": len(data.get("messages", [])),
                })
            except (json.JSONDecodeError, OSError):
                continue
        return sessions

    def load_session(self, session_id: str) -> dict | None:
        filepath = self.sessions_dir / f"{session_id}.json"
        if not filepath.exists():
            return None
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    # ---------- 跨会话检索 ----------

    def find_related(self, query: str, top_k: int = 2) -> list[str]:
        """根据 query 检索相关的历史会话摘要。"""
        if not self._summaries:
            return []

        query_keywords = set(self._extract_keywords(query))
        scored = []

        for item in self._summaries:
            item_keywords = set(item.get("keywords", []))
            overlap = len(query_keywords & item_keywords)
            scored.append((overlap, item["summary"]))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [summary for score, summary in scored[:top_k] if score > 0]

    def _extract_keywords(self, text: str) -> list[str]:
        """简单关键词提取：中文字（2-4 字词）+ 英文单词。"""
        text = text.lower()
        # 中文词语（2-4 字）
        chinese_words = re.findall(r"[一-鿿]{2,4}", text)
        # 英文单词（2+ 字母）
        english_words = re.findall(r"[a-z]{2,}", text)
        return chinese_words + english_words

    def _load_summaries(self) -> list[dict]:
        if not self.summaries_file.exists():
            return []
        try:
            with open(self.summaries_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []

    def _save_summaries(self) -> None:
        with open(self.summaries_file, "w", encoding="utf-8") as f:
            json.dump(self._summaries[-50:], f, ensure_ascii=False, indent=2)  # 只保留最近 50 条

    def _generate_session_id(self) -> str:
        today = datetime.now().strftime("%Y-%m-%d")
        existing = list(self.sessions_dir.glob(f"{today}_*.json"))
        seq = len(existing) + 1
        return f"{today}_{seq:03d}"
