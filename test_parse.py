"""测试工具调用解析逻辑。"""
import json
import sys
sys.path.insert(0, '.')

from src.agent.core import Agent
from src.llm.client import LLMClient
from src.memory.manager import MemoryManager
from src.tools.file_tool import FileTool
from src.tools.code_tool import CodeTool
from src.tools.http_tool import HttpTool

# 模拟 LLM 返回的文本（包含工具调用 JSON）
test_text = """哟，这项目挺杂啊，Electron+Python 混合的？让我瞅瞅你这边到底在捣鼓啥——

{"tool": "file_tool", "arguments": {"operation": "read", "path": "package.json"}}"""

print("测试文本:")
print(repr(test_text))
print()

# 手动执行 _try_parse_tool_call() 的逻辑
text = test_text.strip()
print("1. 去除 Markdown 代码块标记（如果有）")
print(f"   text.startswith('```'): {text.startswith('```')}")
print()

print("2. 检查整段是否是 JSON")
print(f"   text.startswith('{{'): {text.startswith('{')}")
print(f"   text.endswith('}}'): {text.endswith('}')}")
print()

print("3. 从文本中提取第一个 { ... } 块")
start = text.find("{")
end = text.rfind("}")
print(f"   start={start}, end={end}")

if start != -1 and end != -1 and end > start:
    extracted = text[start:end + 1]
    print(f"   extracted: {repr(extracted)}")
    print()
    
    try:
        data = json.loads(extracted)
        print(f"   JSON 解析成功: {data}")
        if "tool" in data and "arguments" in data:
            print("   ✅ 有效的工具调用！")
        else:
            print("   ❌ 缺少 'tool' 或 'arguments' 字段")
    except json.JSONDecodeError as e:
        print(f"   ❌ JSON 解析失败: {e}")
        print()
        print("   诊断: 提取的文本可能不是有效的 JSON")
        print(f"   提取的文本（带不可见字符）: {[c for c in extracted[:50]]}")
else:
    print("   ❌ 未找到 { ... } 块")
