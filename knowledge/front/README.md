# 导入页面

页面为普通 HTML/CSS/JavaScript，无需 npm 或前端构建工具。

前端位于 `knowledge/front`，后端根据 Python 文件位置定位页面，不依赖启动目录。

在 `knowledge` 目录使用 uv 启动：

```powershell
uv sync
uv run uvicorn knowledge.api.main:app --app-dir .. --host 127.0.0.1 --port 8000 --workers 1
```

`--app-dir ..` 用于找到代码中的 `knowledge` 包；页面位置仍由后端自动解析。

在项目根目录启动后端：

```powershell
uv sync --project knowledge
& .\knowledge\.venv\Scripts\python.exe -m uvicorn knowledge.api.main:app --host 127.0.0.1 --port 8000 --workers 1
```

打开 http://127.0.0.1:8000/import。接口文档在 http://127.0.0.1:8000/docs。

保持单 worker，执行导入时不要使用 `--reload`。服务重启后，内存任务不可查询；MinIO 文件和 Milvus 切片仍保留。

独立静态服务部署时，在该页面浏览器控制台设置 `localStorage.setItem('ragApiBase', 'http://127.0.0.1:8000')` 后刷新，并把页面来源加入后端 `CORS_ALLOW_ORIGINS`。不要通过 file:// 双击页面。

PDF 上传包含原件、转换文档及图片归档。MD 只上传单文件，不会自动读取用户电脑的 images 文件夹；资源缺失会显示警告。
