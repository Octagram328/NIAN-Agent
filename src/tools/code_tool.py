"""代码执行工具（Python 沙箱）。"""

import subprocess
import tempfile
import os
from .base import Tool, ToolResult


class CodeTool(Tool):
    """在受控环境中执行 Python 代码，返回 stdout/stderr。"""

    @property
    def name(self) -> str:
        return "code_tool"

    @property
    def description(self) -> str:
        return (
            "执行 Python 代码并返回输出结果。适用于数学计算、数据处理、"
            "格式转换、字符串操作等任务。代码在临时文件中运行，有 10 秒超时限制。"
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "要执行的 Python 代码。注意：不要使用 input()，直接写逻辑和 print() 输出结果。",
                },
            },
            "required": ["code"],
        }

    def execute(self, code: str) -> ToolResult:
        # 简单安全检查：拒绝明显的危险操作
        dangerous = ["__import__", "os.system", "subprocess.call", "subprocess.run",
                     "os.popen", "eval(", "exec(", "open('/etc", "open('C:/Windows"]
        lower_code = code.lower()
        for d in dangerous:
            if d.lower() in lower_code:
                return ToolResult(
                    success=False,
                    content="",
                    error=f"代码包含被禁止的操作: {d}",
                )

        # 创建临时文件执行代码
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
                f.write(code)
                temp_path = f.name

            result = subprocess.run(
                ["python", temp_path],
                capture_output=True,
                text=True,
                timeout=10,
            )

            os.unlink(temp_path)

            output = result.stdout
            if result.stderr:
                output += "\n[stderr]\n" + result.stderr

            if result.returncode != 0:
                return ToolResult(
                    success=False,
                    content=output,
                    error=f"代码执行失败，返回码: {result.returncode}",
                )

            return ToolResult(success=True, content=output.strip())

        except subprocess.TimeoutExpired:
            try:
                os.unlink(temp_path)
            except Exception:
                pass
            return ToolResult(
                success=False,
                content="",
                error="代码执行超时（超过 10 秒）",
            )
        except Exception as e:
            return ToolResult(success=False, content="", error=str(e))
