"""FastAPI 后端：将 Agent 封装为 Web API。"""

import os
import sys
import traceback

# 确保能导入 src 模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from src.agent.core import Agent
from src.llm.client import LLMClient
from src.memory.manager import MemoryManager
from src.tools.file_tool import FileTool
from src.tools.code_tool import CodeTool
from src.tools.http_tool import HttpTool
from src.tools.data_tool import DataTool

load_dotenv(override=True)

app = FastAPI(title="Agent Web GUI")
agent: Agent | None = None


# 全局异常处理器：确保所有未捕获异常都返回 JSON，而不是 HTML
@app.exception_handler(Exception)
async def universal_exception_handler(request, exc):
    print(f"[ERROR] Unhandled exception: {type(exc).__name__}: {exc}")
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"detail": f"{type(exc).__name__}: {str(exc)}"},
    )


@app.on_event("startup")
def startup():
    global agent
    llm = LLMClient()
    tools = [FileTool(), CodeTool(), HttpTool(), DataTool()]
    memory = MemoryManager()
    agent = Agent(llm_client=llm, tools=tools, memory_manager=memory)
    print("[Agent] 已加载，模型:", os.getenv("DEFAULT_MODEL", "unknown"))


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not agent:
        return JSONResponse(
            {"response": "Agent 尚未初始化完成，请稍后再试。"},
            status_code=503,
        )
    try:
        # 每次请求使用独立的 LLMClient，避免连接状态污染
        response = agent.run(req.message)
        return {"response": response}
    except Exception as e:
        # 记录完整异常栈，方便排查
        error_detail = f"{type(e).__name__}: {str(e)}"
        print(f"[ERROR] /chat failed: {error_detail}")
        traceback.print_exc()
        return JSONResponse(
            {"response": f"后端处理出错: {error_detail}"},
            status_code=500,
        )


@app.post("/upload_knowledge")
def upload_knowledge(file: UploadFile = File(...)):
    """上传文档到知识库。"""
    if not agent:
        return JSONResponse({"success": False, "error": "Agent 未初始化"}, status_code=500)

    try:
        # 保存上传文件到临时位置
        temp_path = os.path.join("data", "knowledge", "raw", file.filename)
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        with open(temp_path, "wb") as f:
            f.write(file.file.read())

        count = agent.memory.kb.add_document(temp_path, title=file.filename)
        return {"success": True, "chunks": count, "title": file.filename}
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.get("/memory_status")
def memory_status():
    """获取当前记忆状态（供 UI 右侧面板展示）。"""
    if not agent:
        return {"error": "Agent 未初始化"}

    mm = agent.memory
    return {
        "short_turns": len(mm.short) // 2,
        "mid_summary": mm.mid.summary,
        "kb_docs": len(mm.kb.list_documents()),
        "total_sessions": len(mm.long.list_sessions()),
    }


@app.post("/new_session")
def new_session():
    """开启新会话，清空中短期记忆。"""
    if agent:
        agent.clear_session()
    return {"success": True}


@app.get("/debug_llm")
def debug_llm():
    """诊断端点：直接调用 LLM，绕过 Agent。"""
    if not agent:
        return {"error": "Agent 未初始化"}
    try:
        from src.llm.client import LLMClient
        llm = LLMClient()
        resp = llm.chat('你是助手。简短回复。', [{'role': 'user', 'content': '你好'}])
        return {"ok": True, "response": resp[:100], "len": len(resp)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/")
def root():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))


app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")
