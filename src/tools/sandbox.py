"""代码执行沙箱：AST 静态分析 + 运行时隔离。

提供两层防护：
1. 编译前 AST 检查 —— 拦截危险导入和函数调用
2. 运行时环境隔离 —— 临时目录执行 + 受限 builtins + 文件路径白名单
"""

import ast
import os
import subprocess
import tempfile


# ────────────────────────── 配置 ──────────────────────────

# 禁止导入的模块（及其子模块）
FORBIDDEN_MODULES = {
    "os", "sys", "subprocess", "shutil", "socket", "urllib",
    "http", "ftplib", "smtplib", "poplib", "imaplib", "nntplib",
    "telnetlib", "webbrowser", "importlib", "pkgutil", "pathlib",
    "sqlite3", "dbm", "pickle", "marshal", "shelve", "ctypes",
    "multiprocessing", "threading", "concurrent", "asyncio",
    "tkinter", "idlelib", "turtle", "pywin32",
}

# 允许的标准库模块（不在黑名单即可导入，但白名单用于更严格的提示）
ALLOWED_STDlib = {
    "math", "random", "datetime", "itertools", "collections",
    "statistics", "json", "re", "string", "typing", "fractions",
    "decimal", "hashlib", "base64", "copy", "functools", "operator",
    "textwrap", "unicodedata", "dataclasses", "enum", "numbers",
    "abc", "array", "bisect", "calendar", "cmath", "colorsys",
    "csv", "difflib", "heapq", "pprint", "queue", "time",
    "types", "warnings", "weakref", "zoneinfo", "builtins",
    # 数据科学（如果环境已安装）
    "pandas", "numpy",
}

# 禁止的函数调用（按名称匹配）
FORBIDDEN_CALLS = {
    "eval", "exec", "compile", "__import__",
    "exit", "quit", "globals", "locals", "vars",
    "getattr", "setattr", "delattr",
}

# 禁止的 AST 节点类型
FORBIDDEN_NODES = {"Delete"}


# ────────────────────────── 核心类 ──────────────────────────

class SandboxError(Exception):
    """沙箱安全检查失败"""

    pass


class SandboxExecutor:
    """在隔离环境中执行 Python 代码。

    用法::

        executor = SandboxExecutor(allowed_dirs=["data", "src"])
        ok, stdout, stderr = executor.execute(code)
    """

    def __init__(self, allowed_dirs=None, timeout=10):
        self.timeout = timeout
        # 默认允许当前工作目录
        self.allowed_dirs = [os.path.abspath(d) for d in (allowed_dirs or [os.getcwd()])]

    # ── 第一层：AST 静态分析 ──────────────────────────────

    def check_ast(self, code: str) -> tuple[bool, str]:
        """对代码做 AST 安全检查。

        返回 (通过, 错误信息)
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"语法错误: {e}"

        for node in ast.walk(tree):
            node_type = type(node).__name__

            # 1) 禁止的语法节点
            if node_type in FORBIDDEN_NODES:
                return False, f"禁止的语法: {node_type}（删除操作）"

            # 2) import xxx
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root in FORBIDDEN_MODULES:
                        return False, f"禁止导入模块: {alias.name}"

            # 3) from xxx import yyy
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root = node.module.split(".")[0]
                    if root in FORBIDDEN_MODULES:
                        return False, f"禁止导入模块: {node.module}"

            # 4) 危险函数调用
            elif isinstance(node, ast.Call):
                func_name = self._get_call_name(node.func)
                if func_name and func_name in FORBIDDEN_CALLS:
                    return False, f"禁止调用函数: {func_name}()"

                # 5) os.xxx 系列（即便通过某种方式拿到了 os）
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in ("system", "popen", "remove", "rmdir",
                                          "unlink", "rename", "replace", "kill",
                                          "chmod", "chown"):
                        obj_name = self._get_name(node.func.value)
                        if obj_name in ("os", "_os"):
                            return False, f"禁止调用: {obj_name}.{node.func.attr}()"

        return True, ""

    @staticmethod
    def _get_call_name(node) -> str | None:
        """从 Call.func 中提取函数名。"""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return None

    @staticmethod
    def _get_name(node) -> str | None:
        """提取 Name 节点的 id。"""
        if isinstance(node, ast.Name):
            return node.id
        return None

    # ── 第二层：运行时隔离 ──────────────────────────────

    def _wrap_code(self, code: str, sandbox_dir: str) -> str:
        """在用户代码外包裹沙箱环境。"""
        forbidden_modules_str = ", ".join(f'"{m}"' for m in FORBIDDEN_MODULES)
        # 将反斜杠替换为正斜杠，避免生成的 Python 代码中出现转义序列
        allowed_dirs_str = ", ".join(
            f'"{d.replace(chr(92), "/")}"' for d in self.allowed_dirs + [sandbox_dir]
        )

        wrapper = f'''# ========== 沙箱注入层 ==========
import builtins as _builtins
import os as _os

_ORIG_OPEN = _builtins.open
_ORIG_IMPORT = _builtins.__import__

_ALLOWED_DIRS = [{allowed_dirs_str}]
_FORBIDDEN_MODS = {{{forbidden_modules_str}}}

def _sandbox_open(file, mode="r", *args, **kwargs):
    """受限 open：只允许访问白名单目录。"""
    if isinstance(file, int):          # 文件描述符（如 stdin/stdout）
        return _ORIG_OPEN(file, mode, *args, **kwargs)
    abs_path = _os.path.normpath(_os.path.abspath(str(file)))
    for allowed in _ALLOWED_DIRS:
        norm_allowed = _os.path.normpath(allowed)
        if abs_path.startswith(norm_allowed):
            return _ORIG_OPEN(file, mode, *args, **kwargs)
    raise PermissionError(f"沙箱限制: 无法访问 '{{file}}'")

def _sandbox_import(name, *args, **kwargs):
    """受限 __import__：拦截黑名单模块。"""
    root = name.split(".")[0]
    if root in _FORBIDDEN_MODS:
        raise ImportError(f"沙箱限制: 禁止导入 '{{name}}'")
    return _ORIG_IMPORT(name, *args, **kwargs)

_builtins.open = _sandbox_open
_builtins.__import__ = _sandbox_import

# 尝试屏蔽 os 模块（如果已预加载）
try:
    import os as _preloaded_os
    _preloaded_os.system = lambda *a, **k: (_ for _ in ()).throw(
        PermissionError("os.system 被沙箱禁用")
    )
    _preloaded_os.popen = lambda *a, **k: (_ for _ in ()).throw(
        PermissionError("os.popen 被沙箱禁用")
    )
except Exception:
    pass
# ========== 用户代码开始 ==========
'''
        return wrapper + code

    def execute(self, code: str) -> tuple[bool, str, str]:
        """执行代码，返回 (成功, stdout, stderr_or_error)。"""
        # 1. AST 检查
        ok, err = self.check_ast(code)
        if not ok:
            return False, "", f"[安全检查] {err}"

        # 2. 创建隔离目录
        with tempfile.TemporaryDirectory() as sandbox_dir:
            wrapped = self._wrap_code(code, sandbox_dir)
            script_path = os.path.join(sandbox_dir, "__sandbox__.py")
            with _ORIG_OPEN(script_path, "w", encoding="utf-8") as f:
                f.write(wrapped)

            # 3. 在子进程中执行
            try:
                result = subprocess.run(
                    ["python", script_path],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
                output = result.stdout
                if result.stderr:
                    output += "\n[stderr]\n" + result.stderr

                if result.returncode != 0:
                    err_msg = f"代码执行失败，返回码: {result.returncode}"
                    if result.stderr:
                        err_msg += f"\n{result.stderr.strip()[:500]}"
                    return False, output, err_msg

                return True, output.strip(), ""

            except subprocess.TimeoutExpired:
                return False, "", "代码执行超时（超过 {self.timeout} 秒）"
            except Exception as e:
                return False, "", str(e)


# 在模块内保留原始 open，避免被自己的沙箱误伤
_ORIG_OPEN = open
