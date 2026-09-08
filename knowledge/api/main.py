"""FastAPI 应用装配：创建服务、注册路由、设置跨域和异常响应。

业务入口在 service/import_service.py；HTTP 接口在 api/import_routes.py。
在 knowledge 目录启动：uv run uvicorn knowledge.api.main:app --app-dir ..
"""

import os
from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from knowledge.schema.import_models import ImportSettings
from knowledge.api.import_routes import router
from knowledge.api.upload_middleware import UploadLimitMiddleware
from knowledge.service.import_service import ImportRequestError, ImportService


def create_app(facade_factory=None, runtime_dir=None, max_file_bytes=None):
    """组装应用；可选参数只用于替换测试环境，不让路由关心这些细节。"""
    # 【启动 00】应用创建时校验配置并定义缓存工厂；lifespan 预先创建服务、启动清理任务。
    # 路由的 Depends 通过 dependencies.get_import_service 取得这个实例，不再新建服务。
    # 显式传入 0 也必须触发校验，不能用 “value or default” 悄悄改成默认值。
    overrides = {}
    if runtime_dir is not None:
        overrides["runtime_dir"] = runtime_dir
    if max_file_bytes is not None:
        overrides["max_file_bytes"] = max_file_bytes
    settings = ImportSettings(**overrides)

    @lru_cache(maxsize=1)
    def get_service() -> ImportService:
        """首次创建，之后直接返回缓存实例；每个应用有自己的缓存，避免测试间串数据。"""
        return ImportService(settings, facade=facade_factory() if facade_factory else None)

    @asynccontextmanager
    async def lifespan(app):
        """服务与应用同时启动、关闭；定时清理的具体逻辑由服务自己管理。"""
        # 启动时预先创建服务，之后 Depends 通过同一缓存方法获取。
        service = get_service()
        service.start()
        try:
            yield
        finally:
            try:
                await service.close()
            finally:
                # 服务关闭后不可继续复用，重新启动应用时必须创建新的队列和清理任务。
                get_service.cache_clear()

    app = FastAPI(title="知识库导入服务", lifespan=lifespan)
    app.state.import_service_provider = get_service
    app.include_router(router)
    # 请求体限额属于 HTTP 基础设施；CORS 位于外层，错误响应也包含跨域头。
    app.add_middleware(UploadLimitMiddleware, limit=settings.max_file_bytes + 1024 * 1024)
    origins = os.getenv("CORS_ALLOW_ORIGINS", "http://127.0.0.1:5500,http://localhost:5500,http://127.0.0.1:5173,http://localhost:5173")
    app.add_middleware(CORSMiddleware, allow_origins=[v.strip() for v in origins.split(",") if v.strip()],
                       allow_credentials=False, allow_methods=["GET", "POST"],
                       allow_headers=["Content-Type", "Authorization"], expose_headers=["Retry-After"])

    @app.exception_handler(ImportRequestError)
    async def import_error_handler(request, error):
        """仅在 HTTP 边界把业务错误转换为状态码，service 无需依赖 FastAPI。"""
        status_codes = {"invalid_input": 422, "unsupported_file": 415, "file_too_large": 413,
                        "queue_full": 429, "unavailable": 503, "not_found": 404}
        headers = {"Retry-After": "3"} if error.code == "queue_full" else None
        return JSONResponse({"detail": str(error)}, status_code=status_codes[error.code], headers=headers)

    return app


app = create_app()
