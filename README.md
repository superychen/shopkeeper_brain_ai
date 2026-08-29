# shopkeeper_brain_ai
知识库问答系统。

`knowledge/` 是独立的 Python 子项目；其 `.env`、`pyproject.toml`、`uv.lock`、
`requirements.txt` 和虚拟环境均放在该目录中。

```powershell
# 进入知识库 Python 子项目
Set-Location .\\knowledge

# Windows PowerShell：启用子项目虚拟环境
.\\.venv\\Scripts\\Activate.ps1

# 新增依赖并同步环境
uv add <package>
uv sync
```

主要源码位于 `knowledge/`：`api/`、`processor/`、`prompt/`、`service/`、
`schema/` 和 `test/` 已预先创建。

PDF 导入子流水线位于 `knowledge/processor/import_processor/`：
`main_graph.py` 目前只列出节点编排、边定义与图谱运行的职责，具体流程逻辑与节点实现待后续确定后开发。
