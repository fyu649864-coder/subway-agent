"""FastAPI 应用入口。

启动方式（在 backend 目录下）::

    uvicorn app.main:app --host 0.0.0.0 --port 8000

或::

    python -m app.main
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import chat, health
from app.core.config import settings
from app.core.logging import setup_logging

# 前端静态目录：项目根目录下的 frontend/（backend/app/main.py -> 上溯两级到项目根）
FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时初始化日志。"""

    setup_logging()
    yield


def create_app() -> FastAPI:
    """创建并配置 FastAPI 应用实例。"""

    app = FastAPI(
        title=settings.app_name,
        version=settings.api_version,
        description="地铁乘客情绪智能服务 Agent（当前阶段：项目初始化 + Mock API）",
        lifespan=lifespan,
    )
    # 仅允许配置的前端来源跨域调用（默认本地 8080）。可用环境变量
    # METRO_CORS_ALLOW_ORIGINS 覆盖，逗号分隔多个来源。
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health.router)
    app.include_router(chat.router)

    # 同源托管前端：把 frontend/ 静态文件挂到根路径，使外部用户通过
    # 隧道访问同一域名即可同时获得页面与 API，避免前端写死 127.0.0.1 导致
    # 访问者请求到自己设备的 localhost。路由（/health、/api/v1/chat）已先注册，
    # 优先于静态文件匹配。
    if FRONTEND_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower(),
    )
