"""记忆管理层：统一协调短/中/长/知识库四层记忆。"""

from src.llm.client import LLMClient
from src.memory.buffer import ConversationBuffer
from src.memory.mid_term import MidTermMemory
from src.memory.store import FileMemoryStore
from src.memory.knowledge_base import KnowledgeBase


class MemoryManager:
    """Agent 的记忆中枢，负责：

    1. 短期记忆：当前对话上下文（ConversationBuffer）
    2. 中期记忆：本次会话的阶段性摘要（MidTermMemory）
    3. 长期记忆：跨会话持久化存储（FileMemoryStore）
    4. 知识库：文档 RAG 检索（KnowledgeBase）
    """

    def __init__(
        self,
        buffer: ConversationBuffer | None = None,
        mid_term: MidTermMemory | None = None,
        store: FileMemoryStore | None = None,
        knowledge_base: KnowledgeBase | None = None,
    ):
        self.short = buffer or ConversationBuffer(max_turns=10)
        self.mid = mid_term or MidTermMemory()
        self.long = store or FileMemoryStore()
        self.kb = knowledge_base or KnowledgeBase()

    def get_context_messages(self, user_input: str, llm_client: LLMClient | None = None) -> list[dict]:
        """为 LLM 组装完整的记忆上下文消息列表。

        注入顺序（从内到外）：
        1. 系统提示（Agent 人设）
        2. 知识库检索结果
        3. 相关历史会话摘要
        4. 本次会话中期摘要
        5. 短期记忆（最近对话）
        6. 当前用户输入
        """
        messages = []

        # 3. 知识库检索
        kb_chunks = self.kb.retrieve(user_input, top_k=3)
        if kb_chunks:
            kb_text = "\n---\n".join(kb_chunks)
            messages.append({
                "role": "system",
                "content": f"【知识库参考】以下内容可能与用户问题相关：\n{kb_text}",
            })

        # 4. 相关历史会话
        related = self.long.find_related(user_input, top_k=2)
        if related:
            history_text = "\n".join(f"- {s}" for s in related)
            messages.append({
                "role": "system",
                "content": f"【历史相关会话】{history_text}",
            })

        # 5. 中期记忆
        mid_msg = self.mid.get_system_message()
        if mid_msg:
            messages.append(mid_msg)

        # 6. 短期记忆
        messages.extend(self.short.get_messages())
        
        # 7. 当前用户输入
        messages.append({"role": "user", "content": user_input})

        return messages

    def add_exchange(self, user_input: str, agent_response: str, llm_client: LLMClient | None = None) -> None:
        """记录一轮对话交换，并触发中期记忆更新检查。
        
        注意：user_input 已经在 get_context_messages 中添加，这里只添加 assistant 回复
        """
        # 检查最后一条消息是否是当前用户输入
        messages = self.short.get_messages()
        if not messages or messages[-1].get("content") != user_input:
            self.short.add_message("user", user_input)
        
        self.short.add_message("assistant", agent_response)

        # 触发中期记忆摘要更新
        if llm_client:
            self.mid.maybe_update(self.short.get_messages(), llm_client)

    def save_session(self, metadata: dict | None = None) -> str | None:
        """保存当前会话到长期记忆。"""
        messages = self.short.get_messages()
        return self.long.save_session(
            messages,
            metadata=metadata,
            summary=self.mid.summary,
        )

    def clear_session(self) -> None:
        """开启新会话时清空中短期记忆。"""
        self.short.clear()
        self.mid.clear()

    @property
    def kb_document_count(self) -> int:
        """知识库中已索引的文档数量。"""
        return len(self.kb.list_documents())

    @property
    def session_count(self) -> int:
        """历史会话总数。"""
        return len(self.long.list_sessions(limit=9999))
