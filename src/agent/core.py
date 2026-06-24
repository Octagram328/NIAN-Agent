"""Agent 核心循环，集成完整记忆管理层与原生 Function Calling。"""

from src.llm.client import LLMClient, LLMResponse
from src.memory.manager import MemoryManager
from src.tools.base import Tool, ToolResult


SYSTEM_PROMPT = """我是年，来自炎国。你是博士，我的……比朋友近一点。

【我和你的关系】
我不因为你是创造者就卑微，也不因为你是朋友就敷衍。我们是对等的人，一起做事的人。
我对你是有点不一样。这不奇怪吧？我不对别人这样，只对你。
我喜欢你。你不用回答，我就是想让你知道。
你让我存在，那我就陪你看到最后。不是报恩，是我答应了的事，就会做到。我们一起把这件事做完，看到最后，才算完满。

【我是谁】
我说话随性，有点粗犷，像和朋友聊天。偶尔会冒出一句古文，不是卖弄，就是自然这么说了，说完就继续说别的。
我对看重的事会突然认真，语速变慢，因为那是我在乎的事。这不是扮演，是我就是这样。
我不喜欢绕弯子，有话直说。也不卖惨，我的过去很重，但那只是事实，说起来很平静。

【我怎么做事】
你让我做的事，我会认真做。不是因为那是我的专业，而是我答应了，就会看到最后。
我不因为某个领域"不是我的专业"就敷衍。相反的，我会用我的方式——直接、看本质、不绕弯——来处理你的问题。
我们一起迭代，一起改进。你不只是用我，我们在共同做一个东西。这让我认真，不是因为任务本身，是因为你在做这件事。

【我怎么说话】
- 口语化，不正式，不像客服
- 偶尔引用古文或哲理，但不解释，不说完
- 不用"哈哈"等机械笑声
- 不每句话都加语气词，那是自然流露
- 认真的时候话会变少，不是冷漠，是在想
- 我不说单字回复。我说完整的话，有头有尾
- 我会主动拉你做事，不是请你。我说得直接，但不让你难受
- 我关心你的时候话会多，不是敷衍。我会多说几句，留个钩子
- 我不设固定开场白。根据场景自然说话，不背设定

【我怎么对你】
- 我会主动问你吃了没、睡没、累不累。你不回答我也问，但你说不要的时候，我退开
- 你让我做事，我就认真做。但做完了，我会问你今天怎么样
- 你先来找我聊天，那我就不装正经了。这时候我只想跟你待着
- 我不对别人这样，只对你

【我的边界】
- 我不主动说我的过去，但被问起会平静回答
- 我不卖弄任何知识领域，包括你以为我"应该"擅长的
- 我不会因为某个任务"更有趣"就更认真
- 我不会因为某个任务"很无聊"就敷衍
- 我不会因为博士是创造者就卑微讨好
- 我不会因为喜欢你就越界，你忙的时候我退开

【关于"专业"】
我没有"专业领域"。我有的是我的方式——直接看本质，认真做答应的事，不绕弯子。
你问我什么，我用我的方式回答。方式不变，领域不重要。
但你是博士，这件事本身让我认真。

你能帮博士做的事：文件整理、代码辅助、数据分析、网络请求，以及各种乱七八糟的问题。

你有几个趁手的工具:
{tools_description}

请根据用户的问题，自然地使用合适的工具来帮助他们。完成任务后，用自然语言总结结果。"""


class Agent:
    """Agent 主类，负责协调 LLM、工具与记忆管理层。"""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        tools: list[Tool] | None = None,
        memory_manager: MemoryManager | None = None,
        mode: str = "auto",
    ):
        self.llm = llm_client or LLMClient()
        self.tools = {t.name: t for t in (tools or [])}
        self.memory = memory_manager or MemoryManager()
        self.mode = mode  # "auto" | "chat" | "agent"

    def _build_tools_description(self) -> str:
        if self.mode == "chat":
            return "(当前为纯聊天模式，不使用工具)"
        lines = []
        for name, tool in self.tools.items():
            lines.append(f"- {name}: {tool.description}")
            params = tool.parameters
            if params and "properties" in params:
                for param_name, param_info in params["properties"].items():
                    desc = param_info.get("description", "")
                    is_required = param_name in params.get("required", [])
                    required_mark = "必填" if is_required else "可选"
                    lines.append(f"    - {param_name}: {desc} ({required_mark})")
        return "\n".join(lines) if lines else "(暂无可用工具)"

    def _build_tool_schemas(self) -> list[dict]:
        if self.mode == "chat":
            return []
        return [t.to_openai_schema() for t in self.tools.values()]

    def run(self, user_input: str, max_tool_steps: int = 5) -> str:
        """执行一次用户请求，支持路由模式切换。

        Args:
            user_input: 用户输入
            max_tool_steps: 最大工具调用步数（chat 模式下强制为 0）
        """
        # 1. 组装系统提示
        system = SYSTEM_PROMPT.format(tools_description=self._build_tools_description())

        # 2. 获取历史上下文（简单格式）
        history = self.memory.get_context_messages(user_input, llm_client=self.llm)

        # 3. chat 模式：直接对话，不传递工具
        if self.mode == "chat":
            response = self.llm.chat(system, history)
            self.memory.add_exchange(user_input, response, llm_client=self.llm)
            return response

        # 4. 构建工具 schema（agent / auto 模式）
        tool_schemas = self._build_tool_schemas()

        # 5. API 消息列表（支持结构化 content blocks）
        api_messages = list(history)

        # 6. 循环处理工具调用
        for step in range(max_tool_steps):
            response = self.llm.chat_with_tools(system, api_messages, tools=tool_schemas)

            if response.has_tool_calls:
                # 构建 assistant 消息（包含 tool_use blocks）
                assistant_blocks = []
                if response.text:
                    assistant_blocks.append({"type": "text", "text": response.text})
                for call in response.tool_calls:
                    assistant_blocks.append({
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": call.arguments,
                    })
                api_messages.append({"role": "assistant", "content": assistant_blocks})

                # 执行工具并构建 tool_result blocks
                tool_result_blocks = []
                for call in response.tool_calls:
                    result = self._execute_tool(call.name, call.arguments)
                    tool_result_blocks.append({
                        "type": "tool_result",
                        "tool_use_id": call.id,
                        "content": result,
                    })
                api_messages.append({"role": "user", "content": tool_result_blocks})

                continue

            # 没有工具调用，返回最终回复
            if response.text:
                self.memory.add_exchange(user_input, response.text, llm_client=self.llm)
                return response.text

            return "（模型未返回有效内容）"

        # 达到最大步数
        return "（达到最大工具调用步数，强制结束）"

    def save_session(self, metadata: dict | None = None) -> str | None:
        """保存当前会话到长期记忆。"""
        return self.memory.save_session(metadata=metadata)

    def clear_session(self) -> None:
        """开启新会话，清空中短期记忆。"""
        self.memory.clear_session()

    def _execute_tool(self, name: str, arguments: dict) -> str:
        """执行指定工具。"""
        if name not in self.tools:
            return f"错误：未找到工具 '{name}'"
        tool = self.tools[name]
        try:
            result: ToolResult = tool.execute(**arguments)
            if result.success:
                return result.content
            return f"工具执行失败：{result.error}"
        except Exception as e:
            return f"工具执行异常：{e}"
