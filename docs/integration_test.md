# 后端端到端整合检查与测试

## 一、整合状态

完整处理链路已串通，`/api/v1/chat` 可完成全流程：

```
UserInput
  → PerceptionService（情绪=本地BERT + 事件=规则 + 强度=规则）
  → DecisionService（规则决策引擎 app.decision.rules）
  → ResponseService（本地模板 app.response.templates）
  → ResponseResult
```

各模块均使用本地模型与规则，未调用云端 AI API；证据不足时走保守策略，不编造地铁实时信息、线路编号或预计时间。

经实际运行验证（文本「地铁又延误了我要迟到了！」）：情绪模型正常加载（7 类），链路正确输出 emotion=焦虑、event=地铁延误、strategy=行动建议 及对应回复。

## 二、检查结论

- **数据结构与字段传递**：`emotion_confidence = emotion_score`、`event_confidence = event_score`、`emotion_intensity`、`intensity_evidence`、`event_evidence` 在 Perception → Response → HTTP 响应中全程一致透传，已在端到端测试中断言校验。未发现字段缺失或不一致问题。
- **异常处理**：情绪模型加载/推理失败时，`PerceptionService` 明确抛出 `EmotionModelLoadError`，经 ConversationService 向上传播为 HTTP 500，**不会**静默返回固定或随机情绪。已用注入损坏模型路径的端到端测试验证。
- **本地模型与规则**：确认仅加载 `models/emotion/metro_emotion_v2/`，强制离线（`HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE`）；事件、强度、决策、回复均为本地规则/模板。
- **不编造信息**：保守策略模板不包含具体线路与「预计」类表述，已测试断言。

本次未重复实现已有功能，未做大重构。检查过程中未发现需修复的实际缺陷（现有 45 项测试在整合后仍全部通过）。

## 三、新增测试

新增 `backend/tests/test_e2e_integration.py`（8 个用例），通过 HTTP `/api/v1/chat` 覆盖：

1. 正常对话（延误+赶时间语境）。
2. 不同地铁事件：车厢拥挤、末班车、换乘困难、赶时间。
3. 紧急情况（突发事件冒烟/事故）→ 策略「紧急情况提示」、urgency=high、升级人工、回复含安全提示。
4. 证据不足（「今天天气不错」）→ 事件「其他/未知事件」、策略「保守常规回复」、不编造线路/预计信息。
5. 多轮同一会话。
6. 空文本被 422 拒绝。
7. 情绪模型加载失败 → HTTP 500，不静默降级。
8. 不经 HTTP 的 Python 服务链字段传递校验（投诉场景：升级人工、urgency=high）。

## 四、测试结果

在 `backend/` 目录运行 `python -m pytest tests/ -q`：

```
53 passed, 1 warning in 1.33s
```

（原 45 项 + 新增 8 项端到端，全部通过。warning 为 starlette/httpx 的弃用提示，与本项目无关。）

## 五、遗留问题 / 已知限制

- 回复为预写模板随机选取，未做逐字个性化与多轮上下文融合（本阶段范围外）。
- 模型为 CPU 推理、进程内懒加载单例，首次请求有加载耗时；未做批处理/GPU 优化。
- 情绪模型泛化能力受限于原训练数据，端到端输出的情绪标签以模型实际预测为准，测试不假定具体情绪。
- starlette 对 httpx 的弃用 warning 存在，不影响功能。
