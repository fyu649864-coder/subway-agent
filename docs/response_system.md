# 智能回复系统

基于已有的情绪识别、事件识别、强度分析与规则决策引擎，根据决策结果（strategy）从本地模板库生成自然、共情（empathetic）且实用的中文回复。纯本地模板与规则，不调用云端 API，不重新训练模型。

## 一、新增 / 修改的文件

新增：

- `backend/app/response/templates.py` — 本地回复模板库，按决策策略组织。
- `backend/tests/test_response_system.py` — 回复系统测试。
- `docs/response_system.md` — 本报告。

修改：

- `backend/app/services/response_service.py` — 由原少量硬编码模板改为根据 `decision.strategy` 从模板库渲染回复，保持 `ResponseService.generate(perception, decision)` 接口与 `/api/v1/chat` 完全兼容。

未修改：情绪/事件/强度/决策模块及 API 路由。

## 二、回复策略与模板

模板按决策引擎的 5 种策略组织，每种策略提供多条模板，运行时随机选取以避免机械重复：

| 策略 | 用途 | 倾向 |
| --- | --- | --- |
| 紧急情况提示 | 突发事件 | 安全提示 + 引导求助，优先保障安全 |
| 情绪安抚 | 高强度负面情绪 / 服务投诉 | 共情致歉为主，不编造信息 |
| 行动建议 | 赶时间、延误/拥挤（中低强度）、换乘困难 | 建议关注官方信息与备选方案 |
| 安抚与建议结合 | 延误/拥挤高强度、末班车、中强度情绪 | 先共情再给建议 |
| 保守常规回复 | 证据不足 | 礼貌回应并邀请补充说明，**不编造**实时路况/线路/预计时间 |

未知策略自动回退到保守模板，保证始终有可用输出。

模板均为中文，体现共情（含「理解」「抱歉」「心情」等表达），结合地铁场景。保守策略明确不生成具体线路编号、预计恢复时间等无法核实的信息。

## 三、接入与调用

已接入现有服务链：`/api/v1/chat` → ConversationService → Perception（情绪/事件/强度）→ Decision（规则引擎）→ Response（模板渲染）。响应结构保持兼容，仍返回 `reply`、`template_id` 及全部感知/决策字段。

内部调用：

```python
from app.services.response_service import ResponseService
reply = ResponseService().generate(perception, decision).reply
# 可传 seed=... 使模板选择可复现（便于测试）
```

## 四、测试结果

在 `backend/` 目录运行 `python -m pytest tests/ -q`：

```
45 passed, 1 warning in 0.69s
```

`test_response_system.py` 覆盖：各策略模板可渲染、未知策略回退保守、回复字段完整透传、紧急回复含安全提示、保守策略不编造线路/预计信息、回复含中文共情表达、固定 seed 可复现。全量测试（含既有情绪/事件/强度/决策/API 测试）均通过。

## 五、已知限制

- 回复为预写模板的随机选取，灵活性有限，无法像生成式模型那样逐字个性化；后续可在不调用云端的前提下考虑本地小模型润色。
- 模板不掌握实时地铁数据，行动建议为通用指引（关注官方渠道、备选出行方式），不给出具体线路与时刻，避免编造。
- 回复未做多轮上下文融合（本阶段未开发会话级回复策略），仅基于当前单轮感知与决策结果。
