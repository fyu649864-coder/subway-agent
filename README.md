# 地铁乘客情绪智能服务 Agent

## 一、项目简介

本项目面向地铁乘客在 **延误、拥挤、突发事故、投诉、赶时间** 等场景下产生的
焦虑、愤怒、烦躁、恐惧等负面情绪，提供一个智能对话服务 Agent，对乘客情绪进行
感知、决策并给出共情式回复与行动建议。

## 二、项目目标

- 实时识别乘客文本中的 **情绪类别、情绪强度、事件类型与紧急程度**。
- 基于规则决策选择合适的 **共情 / 行动建议 / 紧急处理** 策略。
- 生成结构化、有温度的回复，缓解乘客负面情绪。
- 保持架构清晰、低耦合，便于后续分阶段迭代。

## 三、技术路线

```
用户输入 → 对话管理器 → 感知层 → 情绪/强度/事件识别
        → 状态结构化 → 规则决策层 → Strategy Manager
        → 共情/行动建议/紧急处理 → 回复模板 → 返回用户
```

- **后端**：Python 3.10+ / FastAPI / Uvicorn / Pydantic
- **感知层**（规划）：BERT / RoBERTa 等中文预训练模型本地推理（PyTorch + Transformers）
- **决策层**（规划）：规则引擎 + Strategy Manager
- **回复层**（规划）：模板回复，后续加入小模型润色
- **前端**（规划）：Vue 3，本阶段仅建立目录占位

> 当前阶段 **不接入云端大语言模型**，感知与决策均为本地 Mock。

## 四、当前项目架构

分层调用关系（严格低耦合，API 不直接调用模型）：

```
API 层 (routes)
   │  依赖注入
   ▼
Conversation Service  ──编排──▶  Perception Service
                                   │
                                   ▼
                              Decision Service
                                   │
                                   ▼
                              Response Service
```

后续接入真实模型时，只需替换对应 Service 内部实现，上层链路无需改动。

## 五、目录结构

```
project/
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI 应用入口
│   │   ├── api/
│   │   │   ├── deps.py              # 依赖注入（服务单例）
│   │   │   └── routes/              # 路由：chat / health
│   │   ├── core/
│   │   │   ├── config.py            # 统一配置管理（pydantic-settings）
│   │   │   └── logging.py           # 日志系统
│   │   ├── models/                  # Pydantic 数据模型（核心数据结构）
│   │   ├── services/                # 服务层：编排感知/决策/回复/对话
│   │   ├── perception/              # 感知层（接口契约，后续接 BERT）
│   │   │   ├── emotion/  event/  intensity/
│   │   ├── decision/                # 决策层（接口契约，后续接规则引擎）
│   │   │   ├── rules/  strategy/
│   │   ├── response/                # 回复层（接口契约，后续接模板/润色）
│   │   │   └── templates/
│   │   └── conversation/            # 对话管理层
│   └── tests/                       # 单元测试
├── frontend/                        # 前端（Vue 3，待开发）
├── data/
│   ├── raw/  processed/  dataset/   # 数据目录
├── training/
│   ├── configs/  scripts/  outputs/ # 模型训练相关
├── models/                          # 训练完成的模型存放
├── docs/                            # 文档
├── requirements.txt
├── README.md
└── .gitignore
```

**核心模块保持独立**：`perception`（感知层）、`decision`（决策层）、
`response`（回复执行层）、`conversation`（对话管理）、`training`（模型训练）、
`models`（训练产物）。

## 六、核心数据结构

定义于 `backend/app/models/`，统一使用 Pydantic，支持向前扩展（默认允许额外字段）。

感知层输出示例：

```json
{
  "text": "2号线怎么还不来，我马上就要迟到了！",
  "emotion": "焦虑",
  "emotion_score": 0.91,
  "intensity": 0.87,
  "event": "地铁延误",
  "event_score": 0.94,
  "urgency": "high"
}
```

已定义模型：`UserInput`、`EmotionResult`、`IntensityResult`、`EventResult`、
`PerceptionResult`、`DecisionResult`、`ResponseResult`、`ConversationState`。

## 七、安装依赖

```bash
# 建议使用虚拟环境
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

> 本阶段仅安装运行时与测试依赖；`torch / transformers / scikit-learn / pandas / numpy`
> 在 `requirements.txt` 中已预留并注释，后续接入模型时再启用。

## 八、启动后端

```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000
# 或
python -m app.main
```

启动后访问交互式文档：http://localhost:8000/docs

## 九、测试 /api/v1/chat

**健康检查：**

```bash
curl http://localhost:8000/health
```

**对话接口：**

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"地铁怎么还不来，我马上要迟到了！\"}"
```

返回（本阶段为 Mock 数据）：

```json
{
  "reply": "非常理解您焦急的心情……",
  "template_id": "tpl_delay_anxiety_high",
  "text": "地铁怎么还不来，我马上要迟到了！",
  "emotion": "焦虑",
  "emotion_score": 0.91,
  "intensity": 0.87,
  "event": "地铁延误",
  "event_score": 0.94,
  "urgency": "high",
  "perception": { "...": "感知层完整结果" },
  "decision": { "...": "决策层完整结果" }
}
```

**运行单元测试：**

```bash
cd backend
pytest -q
```

## 十、配置管理

所有配置集中在 `backend/app/core/config.py`，通过环境变量（前缀 `METRO_`）
或 `.env` 文件覆盖，禁止在业务代码中硬编码。常用项：

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `METRO_API_PORT` | `8000` | API 端口 |
| `METRO_LOG_LEVEL` | `INFO` | 日志级别 |
| `METRO_DEVICE` | `cpu` | 推理设备 cpu/cuda |
| `METRO_MOCK_ENABLED` | `true` | 是否启用 Mock |
| `METRO_MODEL_DIR` | `models` | 模型目录 |
| `METRO_DATASET_PATH` | `data/dataset` | 数据集目录 |

## 十一、日志

记录内容：用户请求、请求时间、感知结果、决策结果、响应结果、异常信息。
遵循最小化原则，**不记录敏感个人信息**（如用户 ID 等）。

## 十二、当前已完成功能

- 清晰、可扩展的项目目录结构
- 统一 Pydantic 核心数据结构（含扩展能力）
- 配置管理（pydantic-settings）与日志系统
- 低耦合的服务层：Conversation / Perception / Decision / Response
- Mock 版 `POST /api/v1/chat` 与 `GET /health`
- 基础单元测试（模型校验 + 接口）

## 十三、后续开发计划

1. **感知层**：接入 BERT / RoBERTa 中文模型，实现情绪 / 强度 / 事件本地推理
2. **决策层**：实现规则引擎与 Strategy Manager
3. **回复层**：模板库管理 + 小模型润色
4. **数据与训练**：数据集构建、模型训练与评估（`training/`、`models/`）
5. **前端**：基于 Vue 3 开发对话界面
6. **工程化**：持久化、部署等

---

*当前阶段：项目初始化 + 基础架构 + 数据结构 + Mock API + 基础测试。*
