# 事件识别 + 情绪强度识别模块

本阶段在现有项目中新增「事件识别」与「情绪强度识别」两个本地规则模块，并与已有的本地情绪分类模型（`models/emotion/metro_emotion_v2/`，BERT 7 分类）整合，通过 `/api/v1/chat` 接口对外提供结构化感知结果。

不训练、不下载模型、不调用云端 API、不开发前端、不开发完整回复系统。

## 一、新增 / 修改的文件

新增：

- `backend/app/perception/emotion_classifier.py` — 本地情绪模型加载与推理封装（懒加载单例、强制离线、加载失败明确抛错）。
- `backend/app/perception/event_recognizer.py` — 规则 + 关键词的事件识别，支持 8 类、多事件、置信度与匹配依据。
- `backend/app/perception/intensity_analyzer.py` — 规则化情绪强度识别（低/中/高），输出档位与判断依据。
- `backend/tests/test_event_intensity.py` — 覆盖 10 类场景的自动化测试。
- `docs/event_and_intensity_module.md` — 本报告。

修改：

- `backend/app/core/config.py` — 新增 `emotion_model_path`（默认指向 `models/emotion/metro_emotion_v2/`）与 `emotion_max_len`，路径统一从配置读取，不硬编码。
- `backend/app/models/perception.py` — `IntensityResult` 增加 `level` / `evidence`；`EventResult` 增加 `evidence` / `all_events`；`PerceptionResult` 增加 `emotion_confidence`、`emotion_intensity`、`intensity_evidence`、`event_confidence`、`event_evidence`（均为可选/带默认值，向后兼容）。
- `backend/app/models/response.py` — `ResponseResult` 同步增加上述字段。
- `backend/app/services/perception_service.py` — 由 Mock 改为真实整合：情绪走本地 BERT 模型，事件与强度走规则模块；模型不可用时明确抛出 `EmotionModelLoadError`，不静默降级。
- `backend/app/services/response_service.py` — 透传新增字段到最终响应。
- `backend/tests/test_chat_api.py` — 原测试假定 Mock 的固定情绪/分值，已改为校验结构合法性（情绪属于 7 类之一、置信度在 [0,1]、字段齐全），不假定模型预测出具体情绪。

## 二、支持的事件类型（8 类）

1. 地铁延误
2. 车厢拥挤
3. 末班车与错过列车
4. 换乘困难
5. 赶时间（赶考试 / 赶飞机 / 赶火车等）
6. 突发事件（车厢或站台）
7. 服务投诉
8. 其他/未知事件（无充分证据时返回，置信度低，不强行猜测）

- 支持常见同义词与口语表达（如「还不来」「挤不上」「没赶上」「绕路」等）。
- 一条文本可命中多个事件，按置信度降序返回，主事件为置信度最高者。
- 返回字段：`event`、`event_confidence`、`event_evidence`（命中的关键词列表）、`all_events`（全部命中明细）。
- 置信度是规则匹配的确定程度（基础 0.6 + 命中词加权 + 强信号加成，上限 0.95），明确标注为规则分数，非训练概率。

## 三、情绪强度如何判断

输出三档：`低` / `中` / `高`，与情绪类别独立判断。

依据（综合打分，不只看感叹号）：

- 高程度词 / 强烈表达：非常、极其、崩溃、气死、受不了、要疯了 等。
- 中程度词：有点、比较、挺、稍微 等。
- 明显情绪词：生气、烦、焦虑、害怕、难过、无助 等。
- 标点：感叹号数量（≥2 加权，单个仅微弱加权）。
- 重复表达：叠词、「！！！」「？？」等。
- 缓和语气：还好、还行、只是、无所谓 等（降低强度）。

规则要点：

- 没有任何情绪/程度关键词的普通陈述，保守判为「中」并附依据说明（不强行判低/高）。
- 空文本或异常输入（None / 非字符串）判为「低」，证据为空。
- 强度判断必须能在无情绪关键词时工作（仅凭程度词与标点也可判高）。
- 返回字段：`emotion_intensity`（档位）、`intensity`（[0,1] 数值，兼容旧字段）、`intensity_evidence`（判断依据）。

## 四、如何调用新模块

内部调用（独立于 API）：

```python
from app.perception.event_recognizer import recognize_events, primary_event
from app.perception.intensity_analyzer import analyze_intensity
from app.perception.emotion_classifier import get_emotion_classifier

hits = recognize_events("地铁又延误了，还赶不上飞机")          # 多事件
event = primary_event("...")                                   # 主事件
intensity = analyze_intensity("这也太慢了吧！！！")            # IntensityResult(level, evidence, score)
emotion, conf, dist = get_emotion_classifier().predict("...")  # 本地模型推理
```

端到端（沿用现有服务链路 API → ConversationService → PerceptionService）：

```python
from app.services.perception_service import PerceptionService
result = PerceptionService().analyze("地铁怎么还不来，我要迟到了！")
# result.emotion / emotion_confidence / emotion_intensity / intensity_evidence
# result.event / event_confidence / event_evidence
```

HTTP 接口：`POST /api/v1/chat`，请求体 `{"text": "..."}`，响应在原有字段基础上新增：
`emotion_confidence`、`emotion_intensity`、`intensity_evidence`、`event_confidence`、`event_evidence`，完整明细见 `perception` / `decision` 内嵌对象。原有字段保持兼容。

模型加载失败行为：`PerceptionService.analyze` 会抛出 `EmotionModelLoadError`，由上层（ConversationService）记录日志并向上传播，**不会**用固定情绪或随机结果代替。

## 五、测试结果

在 `backend/` 目录运行 `python -m pytest tests/ -q`：

```
26 passed, 1 warning in 0.66s
```

覆盖的 10 类场景（见 `backend/tests/test_event_intensity.py`）：

1. 地铁延误 + 焦虑语境 —— 事件「地铁延误」「赶时间」命中，模型正常返回 7 类之一。
2. 车厢拥挤 + 烦躁语境 —— 事件「车厢拥挤」命中。
3. 末班车 + 无助语境 —— 事件「末班车与错过列车」命中，强度 ≥ 中。
4. 赶考试 + 焦虑语境 —— 事件「赶时间」命中。
5. 站台突发事件 —— 事件「突发事件」命中。
6. 服务投诉 + 愤怒语境 —— 事件「服务投诉」命中。
7. 普通陈述、无明确事件 —— 返回「其他/未知事件」，低置信度。
8. 多事件同时出现 —— 延误/拥挤/赶时间同时命中。
9. 空文本或异常输入（None、数字）—— 强度「低」、事件「其他/未知」，不报错。
10. 情绪类别与强度不一致 —— 无情绪词但高程度+重复标点判高；情绪词被缓和语气弱化不判高；无信号保守判中。

另含：本地模型可加载并返回合法 7 类分布、模型加载失败时明确抛错（不静默降级）。

说明：测试**不**假定模型一定预测出指定情绪，事件与强度规则单独验证，模型调用单独验证其返回结构合法。

## 六、已知限制

- 事件识别与情绪强度为**关键词规则**，对隐喻、反讽、复杂指代、网络新梗的覆盖有限；置信度是规则分数，不是模型概率。
- 事件类别集合当前固定 8 类，扩展需在 `event_recognizer._EVENT_KEYWORDS` 中补充关键词。
- 情绪强度为三档离散判定，边界附近文本可能在相邻档位间波动，已通过「无信号保守判中」缓解。
- 情绪模型沿用已训练的 `metro_emotion_v2`，本阶段未重训、未调参，其在未见样本上的泛化能力受限于原训练数据。
- 模型为进程内懒加载单例，首次调用有加载耗时；CPU 推理单条约亚秒级，未做批处理/GPU 优化（本阶段不需要）。
