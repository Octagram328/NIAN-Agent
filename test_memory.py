"""记忆系统快速测试脚本（非交互式）。"""

import os
from dotenv import load_dotenv

from src.llm.client import LLMClient
from src.agent.core import Agent
from src.memory.buffer import ConversationBuffer
from src.memory.store import FileMemoryStore
from src.tools.file_tool import FileTool

load_dotenv(override=True)

print("=" * 50)
print("记忆系统测试")
print("=" * 50)

# 1. 测试 ConversationBuffer
print("\n[1] 测试 ConversationBuffer")
buffer = ConversationBuffer(max_turns=3)
buffer.add_message("user", "你好")
buffer.add_message("assistant", "你好！有什么可以帮你的？")
buffer.add_message("user", "帮我创建一个文件")
buffer.add_message("assistant", "好的，文件名是什么？")
print(f"  当前消息数: {len(buffer)}")
print(f"  消息列表: {buffer.get_messages()}")

# 2. 测试 FileMemoryStore
print("\n[2] 测试 FileMemoryStore")
store = FileMemoryStore()
store.update_preference("theme", "dark")
prefs = store.load_preferences()
print(f"  读取偏好: {prefs}")

session_path = store.save_session(
    [{"role": "user", "content": "测试"}, {"role": "assistant", "content": "收到"}],
    metadata={"test": True}
)
print(f"  会话保存路径: {session_path}")

sessions = store.list_sessions(limit=5)
print(f"  最近会话数: {len(sessions)}")

# 3. 测试 Agent 集成（如果 API Key 存在则调用 LLM，否则跳过）
print("\n[3] 测试 Agent 记忆集成")
llm = LLMClient()
tools = [FileTool()]
agent = Agent(llm_client=llm, tools=tools, buffer=ConversationBuffer(max_turns=5), store=store)

if os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY"):
    print("  调用 LLM 进行第一轮对话...")
    r1 = agent.run("请用一句话自我介绍")
    print(f"  Agent 回复: {r1[:60]}...")
    print(f"  Buffer 消息数: {len(agent.buffer)}")

    print("  调用 LLM 进行第二轮对话（测试上下文记忆）...")
    r2 = agent.run("我刚才让你做什么？")
    print(f"  Agent 回复: {r2[:60]}...")
    print(f"  Buffer 消息数: {len(agent.buffer)}")

    path = agent.save_session(metadata={"test": "integration"})
    print(f"  会话已保存: {path}")
else:
    print("  未检测到 API Key，跳过 LLM 调用测试")

print("\n" + "=" * 50)
print("测试完成")
print("=" * 50)
