"""HTTP 接收层的请求体限额，业务文件校验仍由 service 负责。"""

from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


class UploadLimitMiddleware:
    """在 multipart 解析期间限制实际请求字节数，不能只相信 Content-Length。"""

    def __init__(self, app, limit):
        self.app = app
        self.limit = limit

    async def __call__(self, scope, receive, send):
        """包装 ASGI receive，每收到一块请求体便累加一次。"""
        if scope["type"] != "http" or scope["path"] != "/upload":
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        length = headers.get(b"content-length", b"0")
        if length.isdigit() and int(length) > self.limit:
            return await JSONResponse({"detail": "上传请求过大"}, 413)(scope, receive, send)
        size = 0

        async def limited_receive():
            """计数的是实际接收到的字节，分块传输也执行同一上限。"""
            nonlocal size
            message = await receive()
            size += len(message.get("body", b""))
            if size > self.limit:
                # FastAPI 将此 HTTP 异常转换为 413，提前停止接收过大的 multipart。
                raise StarletteHTTPException(413, "上传请求过大")
            return message
        await self.app(scope, limited_receive, send)


