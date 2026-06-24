"""Embedding 客户端：支持 API 调用 + 本地 Fallback。"""

import os
import math
import re


class EmbeddingClient:
    """统一的文本向量化入口。

    优先使用 OpenAI 兼容的 Embedding API（如 Kimi moonshot-v1-embedding）。
    若 API 不可用，回退到基于词频的简单向量表示（纯 Python，零额外依赖）。
    """

    def __init__(self, api_key: str | None = None, base_url: str | None = None, model: str | None = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        # Embedding 通常走 OpenAI 兼容接口
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL") or "https://api.moonshot.cn/v1"
        self.model = model or os.getenv("EMBEDDING_MODEL", "moonshot-v1-embedding")
        self._client = None
        self._dim = 768  # 简单 fallback 向量的维度

    def _get_client(self):
        if self._client is None:
            import openai
            self._client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def embed(self, texts: list[str]) -> list[list[float]]:
        """将文本列表转换为向量列表。"""
        if not texts:
            return []

        # 尝试 API 调用
        try:
            client = self._get_client()
            response = client.embeddings.create(model=self.model, input=texts)
            return [item.embedding for item in response.data]
        except Exception:
            # API 失败时回退到本地简单向量
            return [self._simple_embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        """单条文本向量化。"""
        return self.embed([text])[0]

    def _simple_embed(self, text: str) -> list[float]:
        """基于字符级 n-gram 频率的简单向量（零依赖 fallback）。"""
        # 提取 2-gram 特征
        text = text.lower()
        features = {}
        for i in range(len(text) - 1):
            bigram = text[i:i + 2]
            features[bigram] = features.get(bigram, 0) + 1

        # 扩展常见中文单字作为补充特征
        for char in text:
            if '一' <= char <= '鿿':
                features[f"c:{char}"] = features.get(f"c:{char}", 0) + 1.5

        # 将特征映射到固定维度向量
        vec = [0.0] * self._dim
        for feat, weight in features.items():
            idx = hash(feat) % self._dim
            vec[idx] += weight

        # L2 归一化
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """计算两个向量的余弦相似度。"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
