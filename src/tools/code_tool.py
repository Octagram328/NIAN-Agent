"""代码执行工具（安全沙箱版）。"""

from .base import Tool, ToolResult
from .sandbox import SandboxExecutor


class CodeTool(Tool):
    """在隔离沙箱中执行 Python 代码，返回 stdout/stderr。"""

    @property
    def name(self) -> str:
        return "code_tool"

    @property
    def description(self) -> str:
        return (
            "执行 Python 代码并返回输出结果。适用于数学计算、数据处理、"
            "格式转换、字符串操作等任务。代码在安全沙箱中运行，有 10 秒超时限制，"
            "禁止文件系统越权访问和危险系统调用。"
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "要执行的 Python 代码。\n"
                        "注意：\n"
                        "- 不要使用 input()\n"
                        "- 禁止导入 os/sys/subprocess/shutil/socket 等系统模块\n"
                        "- 禁止调用 eval/exec/compile/open 等危险函数\n"
                        "- 只允许访问当前工作目录和 data/ 目录下的文件\n"
                        "- 直接写逻辑和 print() 输出结果"
                    ),
                },
            },
            "required": ["code"],
        }

    def execute(self, code: str) -> ToolResult:
        executor = SandboxExecutor(timeout=10)
        ok, stdout, stderr = executor.execute(code)

        if not ok:
            return ToolResult(success=False, content=stdout, error=stderr)

        return ToolResult(success=True, content=stdout)
