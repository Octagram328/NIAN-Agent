"""文件操作工具（MVP 版本）。"""

import os
from .base import Tool, ToolResult


class FileTool(Tool):
    """提供基础文件读写和目录浏览能力。"""

    @property
    def name(self) -> str:
        return "file_tool"

    @property
    def description(self) -> str:
        return (
            "本地文件系统工具。支持以下操作：\n"
            "- read: 读取文件内容\n"
            "- write: 写入或覆盖文件\n"
            "- list: 列出目录内容\n"
            "- append: 追加内容到文件"
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["read", "write", "list", "append"],
                    "description": "要执行的操作类型",
                },
                "path": {
                    "type": "string",
                    "description": "文件或目录路径",
                },
                "content": {
                    "type": "string",
                    "description": "write/append 时使用的内容",
                },
            },
            "required": ["operation", "path"],
        }

    def execute(self, operation: str, path: str, content: str = "") -> ToolResult:
        try:
            if operation == "read":
                if not os.path.isfile(path):
                    return ToolResult(success=False, content="", error=f"文件不存在: {path}")
                with open(path, "r", encoding="utf-8") as f:
                    data = f.read()
                return ToolResult(success=True, content=data)

            elif operation == "write":
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                return ToolResult(success=True, content=f"已写入文件: {path}")

            elif operation == "append":
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                with open(path, "a", encoding="utf-8") as f:
                    f.write(content)
                return ToolResult(success=True, content=f"已追加到文件: {path}")

            elif operation == "list":
                if not os.path.isdir(path):
                    return ToolResult(success=False, content="", error=f"目录不存在: {path}")
                items = os.listdir(path)
                lines = []
                for item in items:
                    full = os.path.join(path, item)
                    prefix = "[D]" if os.path.isdir(full) else "[F]"
                    lines.append(f"{prefix} {item}")
                return ToolResult(success=True, content="\n".join(lines))

            else:
                return ToolResult(success=False, content="", error=f"未知操作: {operation}")

        except Exception as e:
            return ToolResult(success=False, content="", error=str(e))
