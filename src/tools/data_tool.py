"""数据分析工具：支持 CSV/Excel/JSON 的读取、统计描述、查询过滤与可视化。"""

import json
import os

import matplotlib
matplotlib.use("Agg")  # 非交互式后端，避免 GUI 依赖
import matplotlib.pyplot as plt
import pandas as pd

from .base import Tool, ToolResult


class DataTool(Tool):
    """数据分析工具。支持读取表格文件、统计描述、Pandas 查询、生成图表。"""

    @property
    def name(self) -> str:
        return "data_tool"

    @property
    def description(self) -> str:
        return (
            "数据分析工具。支持以下操作：\n"
            "- read: 读取 CSV/Excel/JSON 文件，返回结构信息与预览\n"
            "- describe: 返回数据的统计描述（数值列的 count/mean/std/min/max 等）\n"
            "- query: 用 Pandas 表达式过滤或变换数据，返回结果预览\n"
            "- plot: 生成图表（折线/柱状/散点/直方图）并保存为图片"
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["read", "describe", "query", "plot"],
                    "description": "要执行的操作类型",
                },
                "path": {
                    "type": "string",
                    "description": "数据文件路径（支持 .csv .xlsx .xls .json）",
                },
                "expression": {
                    "type": "string",
                    "description": (
                        "query 时使用的 Pandas 表达式，例如：\n"
                        "  df.query('age > 18')\n"
                        "  df.groupby('category')['sales'].sum().reset_index()\n"
                        "  df.sort_values('date', ascending=False).head(10)"
                    ),
                },
                "plot_type": {
                    "type": "string",
                    "enum": ["line", "bar", "scatter", "hist"],
                    "description": "plot 时的图表类型",
                },
                "x_column": {
                    "type": "string",
                    "description": "plot 时的 X 轴列名",
                },
                "y_column": {
                    "type": "string",
                    "description": "plot 时的 Y 轴列名（hist 不需要）",
                },
                "output_path": {
                    "type": "string",
                    "description": "plot 时图片保存路径（默认保存到 data/plots/）",
                },
            },
            "required": ["operation", "path"],
        }

    # ------------------------------------------------------------------ #
    def execute(
        self,
        operation: str,
        path: str,
        expression: str = "",
        plot_type: str = "line",
        x_column: str = "",
        y_column: str = "",
        output_path: str = "",
    ) -> ToolResult:
        try:
            if operation == "read":
                return self._do_read(path)
            elif operation == "describe":
                return self._do_describe(path)
            elif operation == "query":
                return self._do_query(path, expression)
            elif operation == "plot":
                return self._do_plot(path, plot_type, x_column, y_column, output_path)
            else:
                return ToolResult(success=False, content="", error=f"未知操作: {operation}")
        except Exception as e:
            return ToolResult(success=False, content="", error=str(e))

    # ------------------------------------------------------------------ #
    def _read_df(self, path: str) -> pd.DataFrame:
        """根据扩展名读取文件为 DataFrame。"""
        if not os.path.isfile(path):
            raise FileNotFoundError(f"文件不存在: {path}")

        ext = os.path.splitext(path)[1].lower()
        if ext == ".csv":
            return pd.read_csv(path)
        elif ext in (".xlsx", ".xls"):
            return pd.read_excel(path)
        elif ext == ".json":
            return pd.read_json(path)
        else:
            # 尝试按 CSV 读取，失败再抛异常
            try:
                return pd.read_csv(path)
            except Exception as e:
                raise ValueError(f"不支持的文件格式 '{ext}'，或读取失败: {e}")

    def _do_read(self, path: str) -> ToolResult:
        df = self._read_df(path)
        lines = [
            f"形状: {df.shape[0]} 行 × {df.shape[1]} 列",
            f"列名: {list(df.columns)}",
            "数据类型:",
        ]
        for col, dtype in df.dtypes.items():
            lines.append(f"  {col}: {dtype}")
        lines.append("前 5 行预览:")
        lines.append(df.head(5).to_string(index=False))
        return ToolResult(success=True, content="\n".join(lines))

    def _do_describe(self, path: str) -> ToolResult:
        df = self._read_df(path)
        lines = [
            f"形状: {df.shape[0]} 行 × {df.shape[1]} 列",
            f"列名: {list(df.columns)}",
            "",
            "【数值列统计描述】",
            df.describe().to_string(),
            "",
            "【缺失值统计】",
            df.isnull().sum().to_string(),
        ]
        return ToolResult(success=True, content="\n".join(lines))

    def _do_query(self, path: str, expression: str) -> ToolResult:
        if not expression.strip():
            return ToolResult(success=False, content="", error="query 操作需要提供 expression 参数")

        df = self._read_df(path)

        # 安全检查：禁止危险关键字
        dangerous = ["__import__", "os.system", "subprocess", "eval(", "exec(", "open("]
        lower_expr = expression.lower()
        for d in dangerous:
            if d in lower_expr:
                return ToolResult(success=False, content="", error=f"表达式包含被禁止的操作: {d}")

        # 在受限环境中执行表达式
        # expression 应该是类似 df.query('...') 或 df.groupby(...).sum() 的形式
        local_ns = {"df": df, "pd": pd}
        result = eval(expression, {"__builtins__": {}}, local_ns)

        # 结果可能是 DataFrame 或 Series
        if isinstance(result, pd.DataFrame):
            lines = [
                f"结果形状: {result.shape[0]} 行 × {result.shape[1]} 列",
                result.head(20).to_string(index=False),
            ]
            if result.shape[0] > 20:
                lines.append(f"... （共 {result.shape[0]} 行，仅显示前 20 行）")
            return ToolResult(success=True, content="\n".join(lines))
        elif isinstance(result, pd.Series):
            return ToolResult(success=True, content=result.to_string())
        else:
            return ToolResult(success=True, content=str(result))

    def _do_plot(
        self,
        path: str,
        plot_type: str,
        x_column: str,
        y_column: str,
        output_path: str,
    ) -> ToolResult:
        df = self._read_df(path)

        if not x_column:
            return ToolResult(success=False, content="", error="plot 操作需要提供 x_column 参数")

        if x_column not in df.columns:
            return ToolResult(
                success=False, content="", error=f"X 轴列 '{x_column}' 不存在。可用列: {list(df.columns)}"
            )

        if plot_type != "hist" and y_column and y_column not in df.columns:
            return ToolResult(
                success=False, content="", error=f"Y 轴列 '{y_column}' 不存在。可用列: {list(df.columns)}"
            )

        # 默认输出路径
        if not output_path:
            os.makedirs("data/plots", exist_ok=True)
            base = os.path.splitext(os.path.basename(path))[0]
            output_path = f"data/plots/{base}_{plot_type}.png"

        fig, ax = plt.subplots(figsize=(8, 5))

        try:
            if plot_type == "line":
                if y_column:
                    df.plot(x=x_column, y=y_column, kind="line", ax=ax)
                else:
                    df[x_column].plot(kind="line", ax=ax)
            elif plot_type == "bar":
                if y_column:
                    df.plot(x=x_column, y=y_column, kind="bar", ax=ax)
                else:
                    df[x_column].value_counts().plot(kind="bar", ax=ax)
            elif plot_type == "scatter":
                if not y_column:
                    return ToolResult(success=False, content="", error="scatter 图需要提供 y_column")
                df.plot(x=x_column, y=y_column, kind="scatter", ax=ax)
            elif plot_type == "hist":
                df[x_column].plot(kind="hist", ax=ax, bins=20)
            else:
                return ToolResult(success=False, content="", error=f"不支持的图表类型: {plot_type}")

            ax.set_title(f"{plot_type} - {x_column}" + (f" vs {y_column}" if y_column else ""))
            plt.tight_layout()
            fig.savefig(output_path, dpi=150)
            plt.close(fig)

            return ToolResult(
                success=True,
                content=f"图表已保存: {output_path}\n类型: {plot_type}\nX: {x_column}" + (f"\nY: {y_column}" if y_column else ""),
            )
        except Exception as e:
            plt.close(fig)
            return ToolResult(success=False, content="", error=f"绘图失败: {e}")
