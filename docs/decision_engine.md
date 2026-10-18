# 规则决策引擎

基于已有的情绪识别、事件识别与强度分析结果，制定本地规则决定回复策略，为后续回复系统提供结构化决策结果。纯规则实现，不训练模型、不调用云端 API。

## 一、新增 / 修改的文件

新增：

- `backend/app/decision/rules.py` — 规则决策引擎（规则库 + 匹配逻辑）。
- `backend/tests/test_decision_engine.py` — 决策引擎测试。
- `docs/decision_engine.md` — 本报告。

修改：

- `backend/app/services/decision_service.py` — 由原 Mock 硬编码改为委托 `app.decision.rules.evaluate`，保留 `DecisionService.decide` 接口不变，服务调用链（API → ConversationService → Perception → Decision → Response）与 `/api/v1/chat` 完全兼容。

## 二、支持的回复策略

1. **紧急情况提示**（`紧急情况提示`）— 突发事件，安全优先，升级人工介入。
2. **情绪安抚**（`情绪安抚`）— 高强度负面情绪 / 服务投诉，共情为主。
3. **行动建议**（`行动建议`）— 赶时间、延误（中低强度）、拥挤（中低强度）、换乘困难等，以信息告知与可执行建议为主。
4. **安抚与建议结合**（`安抚与建议结合`）— 延误/拥挤高强度、末班车、中强度情绪，先共情再给建议。
5. **保守常规回复**（`保守常规回复`）— 证据不足（事件未知且情绪平静）时采用，避免过度承诺。

## 三、规则设计

- 规则为 `Rule` 数据结构，匹配字段 `event / emotion / intensity` 支持通配（None），`priority` 越小越优先。
- 匹配顺序：突发事件安全类 → 明确事件+强度 → 仅事件 → 仅强度/情绪 → 保守默认。
- 场景覆盖：地铁延误（分高/中低强度）、车厢拥挤（分高/中低强度）、末班车与错过列车、换乘困难、赶时间、突发事件、服务投诉。
- 输出 `DecisionResult`：`strategy`、`actions`（动作列表）、`urgency`、`needs_escalation`、`rationale`、`extras.matched_rule`（命中规则名，便于审计）。
- 突发事件与服务投诉强制 `urgency=high` 且 `needs_escalation=True`。
- 扩展性：新增场景只需在 `RULES` 追加一条规则，无需改动匹配逻辑。

## 四、如何调用

内部：

```python
from app.decision import rules
from app.models import PerceptionResult

decision = rules.evaluate(perception)  # perception 来自感知层
# decision.strategy / actions / urgency / needs_escalation / rationale
```

或经服务链：`DecisionService().decide(perception)`，与现有 `/api/v1/chat` 自动整合。

## 五、测试结果

在 `backend/` 目录运行 `python -m pytest tests/ -q`：

```
38 passed, 1 warning in 0.68s
```

`test_decision_engine.py` 覆盖：突发事件升级、延误高/低强度、拥挤、末班车、赶时间、换乘困难、服务投诉、证据不足保守策略、无事件高强度安抚、决策字段完整性，以及全场景批量合法性校验。所有用例直接构造 `PerceptionResult`，不依赖情绪模型推理，确定性验证规则匹配。

## 六、已知限制

- 规则为人工编写的关键词/类别匹配，无法覆盖所有口语与隐喻表达；置信度与细分边界依赖上游感知质量。
- 强度分档仅三档，相邻档位文本可能在不同规则间波动。
- 策略与动作模板为文本标签，具体回复文案仍由回复层（ResponseService）消费 `actions` 生成，本阶段未开发完整回复系统。
- 规则优先级为静态设定，未引入学习/自动调优机制。
