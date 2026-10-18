"""健康检查路由：GET /health。"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["system"])


@router.get("/health", summary="健康检查")
def health() -> dict[str, str]:
    return {"status": "ok"}
