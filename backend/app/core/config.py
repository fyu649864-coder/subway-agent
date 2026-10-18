"""统一配置管理。

所有可配置项集中在此处，业务代码通过 ``from app.core.config import settings``
读取，严禁在业务代码中硬编码路径、端口、设备等参数。

配置来源优先级（pydantic-settings）：
    环境变量 > .env 文件 > 代码默认值

环境变量统一使用 ``METRO_`` 前缀，例如 ``METRO_API_PORT=9000``。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用全局配置。"""

    model_config = SettingsConfigDict(
        env_prefix="METRO_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- 应用基本信息 ----
    app_name: str = Field(default="地铁乘客情绪智能服务 Agent", description="应用名称")
    api_version: str = Field(default="v1", description="API 版本")

    # ---- 服务运行 ----
    api_host: str = Field(default="0.0.0.0", description="API 监听地址")
    api_port: int = Field(default=8000, description="API 监听端口")
    log_level: str = Field(default="INFO", description="日志级别")

    # ---- CORS ----
    # 仅允许实际需要的前端地址，避免任意来源跨域调用。
    # 多个来源用逗号分隔，例如 METRO_CORS_ALLOW_ORIGINS=http://127.0.0.1:8080,http://localhost:8080
    cors_allow_origins: str = Field(
        default="http://127.0.0.1:8080,http://localhost:8080",
        description="允许跨域的前端来源，逗号分隔",
    )

    # ---- 运行模式 ----
    mock_enabled: bool = Field(default=True, description="是否启用 Mock（本阶段为 True）")

    # ---- 推理设备 ----
    device: str = Field(default="cpu", description="推理设备：cpu / cuda")

    # ---- 路径配置 ----
    model_dir: Path = Field(default=Path("models"), description="训练完成模型存放目录")
    data_dir: Path = Field(default=Path("data"), description="数据根目录")
    dataset_path: Path = Field(default=Path("data/dataset"), description="数据集目录")

    # ---- 模型路径 ----
    # 默认指向项目根目录下的本地情绪分类模型（v3，7 类）。
    # 旧模型 v2 目录完整保留，可随时改回此路径回滚。业务代码不硬编码路径，统一从 settings 读取。
    emotion_model_path: Optional[str] = Field(
        default=str(Path(__file__).resolve().parents[3] / "models" / "emotion" / "metro_emotion_v3"),
        description="情绪识别模型路径（本地目录）",
    )
    event_model_path: Optional[str] = Field(default=None, description="事件识别模型路径（本阶段使用规则，暂不使用）")
    intensity_model_path: Optional[str] = Field(default=None, description="强度识别模型路径（本阶段使用规则，暂不使用）")

    # ---- 情绪模型推理 ----
    emotion_max_len: int = Field(default=64, description="情绪模型最大序列长度")

    # ---- 对话管理 ----
    max_history: int = Field(default=20, description="单会话保留的最大历史轮数")

    # ---- 本地大模型回复生成（可选）----
    # 启用后，非紧急回复优先由本地 LLM 生成；失败/超时/被禁用时自动回退模板。
    # 紧急安全策略始终使用模板，规则优先级高于大模型。
    llm_enabled: bool = Field(default=True, description="是否启用本地大模型生成回复")
    llm_model_path: str = Field(
        default=str(Path(__file__).resolve().parents[3] / "models" / "llm" / "qwen2.5-3b-instruct"),
        description="本地中文指令模型目录（transformers 格式）",
    )
    llm_max_new_tokens: int = Field(default=100, description="LLM 单次最大生成 token 数")
    llm_timeout: float = Field(default=45.0, description="LLM 生成超时（秒），超时回退模板")
    llm_temperature: float = Field(default=0.7, description="生成温度")
    llm_device: str = Field(default="cpu", description="LLM 推理设备 cpu/cuda")

    @property
    def cors_origins(self) -> list[str]:
        """解析逗号分隔的允许来源列表，去空白与空项。"""
        return [o.strip() for o in self.cors_allow_origins.split(",") if o.strip()]


# 全局单例，业务代码统一引用此对象
settings = Settings()
