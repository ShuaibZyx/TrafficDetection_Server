from fastapi import Request
from fastapi.responses import JSONResponse
from loguru import logger
from src.template.response import response_error


def register_global_exception(app):
    """
    注册全局异常处理器
    """

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, e: Exception):
        logger.exception(f"请求 {request.url} 出现异常: {e}")
        response = JSONResponse(
            status_code=500, content=response_error(msg=str(e)).model_dump()
        )
        return response
