# 前端聊天界面

## 一、检查结论

- 现有 `frontend/` 目录仅含 `.gitkeep`，无任何既有前端代码，因此新建了一套轻量静态前端，未重复搭建后端。
- 后端 `/api/v1/chat` 实际响应结构（已通过真实请求抓取确认）包含：`reply`、`template_id`、`emotion`、`emotion_score`/`emotion_confidence`、`intensity`、`emotion_intensity`、`intensity_evidence`、`event`、`event_score`/`event_confidence`、`event_evidence`、`urgency`，以及内嵌 `perception`、`decision`（含 `strategy`、`actions`、`needs_escalation`）。前端按这些实际字段渲染，缺失字段做兼容（显示「未知」或隐藏），不虚构识别结果。
- 后端原本未启用 CORS，浏览器跨域调用会被拦截，故在 `backend/app/main.py` 增加了 CORSMiddleware（开发阶段 `allow_origins=["*"]`），未改动任何业务模块。

## 二、新增 / 修改文件

新增（`frontend/`）：

- `index.html` — 页面结构：头部（连接状态）、消息区、输入框 + 发送按钮。
- `styles.css` — 简洁美观的桌面端样式，自适应常见屏幕尺寸（含 ≤600px 适配）。
- `app.js` — 原生 JS，无构建依赖：健康检查、发送消息、加载动画、错误提示、渲染情绪/强度/事件/策略/建议分析面板，会话 ID 存 localStorage。

修改：

- `backend/app/main.py` — 增加 CORS 中间件（仅新增，不影响现有逻辑与测试，`53 passed` 保持通过）。

## 三、功能

- 用户输入、回车或点击发送、消息气泡展示、发送中 loading 动画、请求失败错误提示。
- 顶部自动检测后端 `/health`，显示「后端已连接 / 未连接」，未连接时禁用发送。
- 每条回复下方渲染分析面板：情绪（含置信度）、情绪强度（高标红）及依据、事件（含置信度，突发事件标红）、策略、建议动作列表；`needs_escalation` 时显示红色「已升级人工」标签。
- 缺失字段兼容处理，绝不显示编造的识别结果；界面提示实时信息以官方渠道为准。

## 四、运行方式

1. 启动后端：
   ```
   cd backend
   python -m uvicorn app.main:app --port 8000
   ```
2. 启动前端（任选其一）：
   ```
   cd frontend
   python -m http.server 8080 --bind 127.0.0.1
   ```
   浏览器打开 http://127.0.0.1:8080/ 。
   （若后端地址不同，可用 `http://127.0.0.1:8080/?api=http://host:port` 指定。）

## 五、测试结果

- 后端单元/集成测试：`python -m pytest tests/ -q` → **53 passed**（含 CORS 改动后回归，未破坏既有功能）。
- CORS 预检实测：`OPTIONS /api/v1/chat` 返回 200，`Access-Control-Allow-Origin: *` 正常。
- 浏览器端到端实测（http://127.0.0.1:8080/，真实调用后端）：
  - 「地铁又延误了，我快要迟到了！」→ 成功返回并展示：情绪=焦虑 95%、强度=高、事件=地铁延误 75%、策略=安抚与建议结合、建议列表正常。
  - 「前面车厢好像冒烟了，是不是出事故了」→ 事件=突发事件，策略旁出现红色「已升级人工」标签，回复含「注意安全」等安全提示。
  - 控制台无本次相关 JS/API 异常。

## 六、遗留问题 / 已知限制

- 前端为无构建的静态页面，未做用户登录、历史消息持久化（仅会话 ID 本地保存，刷新后消息清空）。
- CORS 当前为开发态全开放，生产部署需收紧 `allow_origins`。
- 回复为后端模板，前端只负责展示，不做内容生成。
- 未接入前端自动化测试框架（浏览器手工/Agent 实测通过）。
