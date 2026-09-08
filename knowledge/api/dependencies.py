"""FastAPI 依赖注入：把应用托管的服务交给路由使用。

main.py 的 get_service 使用 lru_cache 缓存实例，lifespan 负责启动、关闭和清空缓存。
路由用 Depends(get_import_service) 声明依赖，作用类似 Spring 注入已有的 Bean。
"""

from typing import cast

from fastapi import Request

from knowledge.service.import_service import ImportService


async def get_import_service(request: Request) -> ImportService:
    """返回当前应用共用的导入服务，保证上传和轮询使用同一份任务队列与状态。

    app.state 的存放细节集中在这里，路由只需要知道具体的 ImportService 类型。
    缓存放在无参数的服务工厂上，不放在本方法上，避免把每个 Request 保留在缓存里。
    cast 只给编辑器提供明确的服务类型。
    """
    # 【流程 02 · 注入】上传和查询共用当前应用的缓存服务，因此能查到后台更新的任务。
    # 取到实例后返回 import_routes.upload/status，继续执行接口方法。
    return cast(ImportService, request.app.state.import_service_provider())
