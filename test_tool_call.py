"""测试工具调用功能"""

import sys
import os
sys.path.insert(0, r'D:\nian')

# 加载环境变量
from dotenv import load_dotenv
load_dotenv(r'D:\nian\.env')

from src.agent.core import Agent
from src.tools.file_tool import FileTool

def test_tool_call():
    print("=" * 50)
    print("测试工具调用功能")
    print("=" * 50)
    
    # 创建 Agent
    agent = Agent(tools=[FileTool()])
    
    # 测试 1: 不需要工具的对话
    print("\n【测试 1】不需要工具的对话")
    print("-" * 50)
    response = agent.run("你好，请介绍一下自己")
    print(f"用户: 你好，请介绍一下自己")
    print(f"Agent: {response}")
    
    # 测试 2: 单步工具调用
    print("\n【测试 2】单步工具调用 - 列出当前目录")
    print("-" * 50)
    response = agent.run("帮我列出当前目录的文件")
    print(f"用户: 帮我列出当前目录的文件")
    print(f"Agent: {response}")
    
    # 测试 3: 多步工具调用（如果上一步成功）
    print("\n【测试 3】读取特定文件")
    print("-" * 50)
    response = agent.run("帮我读取 README.md 文件的内容")
    print(f"用户: 帮我读取 README.md 文件的内容")
    print(f"Agent: {response}")
    
    print("\n" + "=" * 50)
    print("测试完成")
    print("=" * 50)

if __name__ == "__main__":
    test_tool_call()
