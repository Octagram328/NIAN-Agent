"""HTTP 网络请求工具。"""

import requests
from .base import Tool, ToolResult


class HttpTool(Tool):
    """发送 HTTP 请求（GET/POST/PUT/DELETE），返回响应内容。"""

    @property
    def name(self) -> str:
        return "http_tool"

    @property
    def description(self) -> str:
        return (
            "发送 HTTP 请求获取网页内容或调用 API。支持 GET/POST/PUT/DELETE。"
            "GET 用于获取数据，POST 用于提交数据，PUT 用于更新，DELETE 用于删除。"
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "请求的目标 URL，例如 https://api.example.com/data",
                },
                "method": {
                    "type": "string",
                    "enum": ["GET", "POST", "PUT", "DELETE"],
                    "description": "HTTP 方法，默认为 GET",
                },
                "headers": {
                    "type": "object",
                    "description": "可选的请求头，例如 {'Content-Type': 'application/json'}",
                },
                "body": {
                    "type": "string",
                    "description": "请求体内容（POST/PUT 时使用），JSON 数据请作为字符串传入",
                },
            },
            "required": ["url", "method"],
        }

    def execute(self, url: str, method: str = "GET", headers: dict | None = None, body: str = "") -> ToolResult:
        try:
            req_headers = headers or {}
            kwargs = {"url": url, "headers": req_headers, "timeout": 15}

            if body and method in ("POST", "PUT"):
                kwargs["data"] = body

            response = requests.request(method.upper(), **kwargs)

            # 截断过长的响应
            content = response.text
            if len(content) > 4000:
                content = content[:4000] + "\n... [内容已截断，超过 4000 字符]"

            result_text = f"Status: {response.status_code}\nContent:\n{content}"
            return ToolResult(success=True, content=result_text)

        except requests.Timeout:
            return ToolResult(success=False, content="", error="请求超时（超过 15 秒）")
        except requests.RequestException as e:
            return ToolResult(success=False, content="", error=f"请求失败: {e}")
        except Exception as e:
            return ToolResult(success=False, content="", error=str(e))
