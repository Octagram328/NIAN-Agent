"""知识库 RAG：文档上传、切分、向量化存储、语义检索。"""

import json
import os
import re
from pathlib import Path
from typing import Any

from src.llm.embeddings import EmbeddingClient, cosine_similarity


class KnowledgeBase:
    """本地知识库，支持文档上传和语义检索。

    存储结构：
        data/
        └── knowledge/
            ├── index.json      # 文档切分后的 chunks + 向量
            └── raw/            # 原始上传文件（可选保留）
    """

    def __init__(self, data_dir: str | None = None):
        self.data_dir = Path(data_dir or "data") / "knowledge"
        self.raw_dir = self.data_dir / "raw"
        self.index_file = self.data_dir / "index.json"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        self._index: list[dict[str, Any]] = []
        self._embedding_client = EmbeddingClient()
        self._load_index()

    # ---------- 文档管理 ----------

    def add_document(self, filepath: str, title: str | None = None) -> int:
        """上传文档并建立索引，返回切分后的 chunk 数量。"""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"文件不存在: {filepath}")

        text = self._extract_text(filepath)
        if not text.strip():
            raise ValueError("文件内容为空或无法提取文本")

        title = title or filepath.name
        chunks = self._split_text(text)

        # 获取向量
        embeddings = self._embedding_client.embed(chunks)

        # 添加到索引
        for chunk, emb in zip(chunks, embeddings):
            self._index.append({
                "title": title,
                "source": str(filepath),
                "content": chunk,
                "embedding": emb,
            })

        self._save_index()
        return len(chunks)

    def add_text(self, text: str, title: str) -> int:
        """直接添加文本内容。"""
        chunks = self._split_text(text)
        embeddings = self._embedding_client.embed(chunks)
        for chunk, emb in zip(chunks, embeddings):
            self._index.append({
                "title": title,
                "source": f"inline:{title}",
                "content": chunk,
                "embedding": emb,
            })
        self._save_index()
        return len(chunks)

    def list_documents(self) -> list[str]:
        """列出已索引的文档标题（去重）。"""
        titles = set()
        for item in self._index:
            titles.add(item["title"])
        return sorted(titles)

    def clear(self) -> None:
        """清空知识库。"""
        self._index.clear()
        if self.index_file.exists():
            self.index_file.unlink()

    # ---------- 检索 ----------

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        """根据 query 检索最相关的知识片段。"""
        if not self._index:
            return []

        query_emb = self._embedding_client.embed_query(query)

        scored = []
        for item in self._index:
            sim = cosine_similarity(query_emb, item["embedding"])
            scored.append((sim, item["content"]))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [content for _, content in scored[:top_k]]

    # ---------- 内部方法 ----------

    def _extract_text(self, filepath: Path) -> str:
        """从文件提取纯文本。"""
        suffix = filepath.suffix.lower()

        if suffix in (".txt", ".md", ".py", ".json", ".csv", ".log"):
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()

        if suffix == ".pdf":
            try:
                import PyPDF2
                reader = PyPDF2.PdfReader(str(filepath))
                return "\n".join(page.extract_text() or "" for page in reader.pages)
            except Exception:
                return ""

        # 未知格式尝试按文本读取
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            return ""

    def _split_text(self, text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
        """将文本切分为重叠的 chunks。"""
        # 先按段落粗分
        paragraphs = re.split(r"\n\s*\n", text)
        chunks = []
        current = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(current) + len(para) < chunk_size:
                current += para + "\n\n"
            else:
                if current:
                    chunks.append(current.strip())
                current = para + "\n\n"

        if current:
            chunks.append(current.strip())

        # 如果单段过长，再强制按字符切分
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= chunk_size:
                final_chunks.append(chunk)
            else:
                start = 0
                while start < len(chunk):
                    end = start + chunk_size
                    final_chunks.append(chunk[start:end])
                    start = end - overlap

        return final_chunks

    def _load_index(self) -> None:
        """加载索引文件。"""
        if not self.index_file.exists():
            return
        try:
            with open(self.index_file, "r", encoding="utf-8") as f:
                self._index = json.load(f)
        except (json.JSONDecodeError, OSError):
            self._index = []

    def _save_index(self) -> None:
        """保存索引文件。"""
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(self._index, f, ensure_ascii=False, indent=2)
