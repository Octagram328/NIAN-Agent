"""Agent 入口脚本。"""

import os
from dotenv import load_dotenv

from src.llm.client import LLMClient
from src.agent.core import Agent
from src.memory.manager import MemoryManager
from src.tools.file_tool import FileTool
from src.tools.code_tool import CodeTool
from src.tools.http_tool import HttpTool
from src.tools.data_tool import DataTool


def main():
    load_dotenv(override=True)

    # 检查 API Key
    provider = os.getenv("LLM_PROVIDER", "anthropic")
    if provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        print("请设置 ANTHROPIC_API_KEY 环境变量，或复制 .env.example 为 .env 并填写密钥")
        return
    if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        print("请设置 OPENAI_API_KEY 环境变量")
        return

    # 初始化 Agent（MemoryManager 在内部自动创建）
    llm = LLMClient()
    tools = [FileTool(), CodeTool(), HttpTool(), DataTool()]
    agent = Agent(llm_client=llm, tools=tools)

    # 显示当前配置
    model = os.getenv("DEFAULT_MODEL", "unknown")
    base_url = os.getenv("ANTHROPIC_BASE_URL", "") if provider == "anthropic" else os.getenv("OPENAI_BASE_URL", "")
    sessions = agent.memory.long.list_sessions(limit=3)
    print("=" * 50)
    print("[Agent] 已启动（完整记忆层）")
    print(f"   Provider : {provider}")
    print(f"   Model    : {model}")
    if base_url:
        print(f"   Base URL : {base_url}")
    if sessions:
        print(f"   历史会话 : {len(sessions)} 条")
    print(f"   知识库   : {agent.memory.kb_document_count} 个文档")
    print("-" * 50)
    print("支持命令：")
    print("  /exit  - 退出并保存当前会话")
    print("  /new   - 开启新会话（清空当前上下文）")
    print("  /save  - 手动保存当前会话")
    print("  /tools - 查看可用工具")
    print("  /kb    - 查看知识库文档")
    print("=" * 50)

    while True:
        try:
            user_input = input("\n你: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见！")
            break

        if not user_input:
            continue
        if user_input == "/exit":
            break
        if user_input == "/new":
            agent.clear_session()
            print("[系统] 已开启新会话，上下文已清空。")
            continue
        if user_input == "/save":
            path = agent.save_session(metadata={"source": "manual_save"})
            if path:
                print(f"[系统] 会话已保存: {path}")
            else:
                print("[系统] 保存失败。")
            continue
        if user_input == "/tools":
            for t in tools:
                print(f"  - {t.name}: {t.description}")
            continue
        if user_input == "/kb":
            docs = agent.memory.kb.list_documents()
            if docs:
                print(f"[知识库] {len(docs)} 个文档:")
                for d in docs:
                    print(f"  - {d}")
            else:
                print("[知识库] 暂无文档")
            continue

        print("\nAgent 思考中...")
        response = agent.run(user_input)
        print(f"\nAgent: {response}")

    # 退出时自动保存
    path = agent.save_session(metadata={"source": "auto_save_on_exit"})
    if path:
        print(f"[系统] 会话已自动保存: {path}")


if __name__ == "__main__":
    main()
