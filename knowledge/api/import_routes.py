"""HTTP 接口只做三件事：接收参数、调用服务、返回结果。

文件校验、落盘、容量检查、任务提交及状态查找均在 ImportService 中。
同步 def 上传接口由 FastAPI 自动放入线程池，慢文件读写不会堵塞事件循环。

全链路阅读顺序（跨文件的【流程】编号一致；函数内 Step 仅表示局部步骤）：
    启动 00：main.create_app 缓存并托管 ImportService。
    01：front/import.html 上传 → 02：本文件 upload → 03：ImportService.upload。
    04：ImportTaskFacade.submit 入队，_work 取任务 → 05：_run 备份原件、准备副本。
    06：main_graph.run_import_graph → 07.1～07.7：检查、PDF 转换、图片、切片、商品名、编码、入库。
    08：ImportArtifactService.finish 归档产物 → 09：TaskStore.finish 提交 completed。
    10：页面 poll → 本文件 status → ImportService.get_status → TaskStore.snapshot。
流程 10 与后台执行并行；异常先提交 failed，再尽力备份失败清单。
"""

from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, Response, UploadFile
from fastapi.responses import FileResponse

from knowledge.api.import_schemas import TaskStatusResponse, UploadAccepted
from knowledge.api.dependencies import get_import_service
from knowledge.service.import_service import ImportService

router = APIRouter()
FRONT_DIR = Path(__file__).resolve().parents[1] / "front"


@router.get("/", include_in_schema=False)
@router.get("/import", include_in_schema=False)
def page():
    """按模块所在位置定位页面，避免 uv 启动目录影响前端访问。"""
    # 【流程 01 · 页面入口】访问项目根地址或 /import，返回 front/import.html。
    # 页面 upload() 发送文件到下面的 POST /upload。
    return FileResponse(FRONT_DIR / "import.html")


@router.post("/upload", status_code=202, response_model=UploadAccepted)
def upload(
    file: UploadFile = File(...),
    document_id: str = Form(""),
    service: ImportService = Depends(get_import_service),
):
    """FastAPI 自动注入 ImportService；文件流在请求结束时由框架关闭。"""
    # 【流程 02 · 上传接口】FastAPI 解析表单并注入服务后，将文件流交给 ImportService.upload（流程 03）。
    # 这里返回的 202 只表示已接纳；实际导入由后台队列执行。
    return service.upload(file.filename, file.file, document_id)


@router.get("/status/{task_id}", response_model=TaskStatusResponse)
async def status(
    response: Response,
    task_id: str,
    service: ImportService = Depends(get_import_service),
):
    """查询只读取轻量内存快照，禁止浏览器缓存旧进度。"""
    # 【流程 10 · 查询接口】与后台导入并行执行，只调用 get_status 取快照，不触发图或数据库操作。
    response.headers["Cache-Control"] = "no-store"
    return service.get_status(task_id)
