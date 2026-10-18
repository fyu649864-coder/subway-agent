"""本地大模型回复生成器。

使用 transformers 加载本地中文指令模型（默认 Qwen2.5-3B-Instruct），
为非紧急对话生成自然、有共情、能承接上下文的回复。

设计约束：
- 纯本地推理，不调用任何云端 API。
- 紧急安全策略不经过本模块，由模板层处理（规则优先级高于大模型）。
- 模型缺失、加载失败、生成超时或输出不合格时返回 ``None``，
  调用方据此回退到现有模板，保证服务始终可用。
- 系统提示硬约束：不得编造地铁线路、列车状态、预计等待时间等未经确认的事实。
"""

from __future__ import annotations

import threading
import time
from typing import Optional

from app.core.config import settings
from app.core.logging import get_logger
from app.models import DecisionResult, PerceptionResult
from app.response import games
from app.response.context import ReplyContext, extract_context

logger = get_logger(__name__)

_SYSTEM_PROMPT = """你是一个愿意倾听、陪着乘客说话的朋友，正在地铁里通过聊天陪伴一位乘客。请用自然、口语化、温和真诚的中文回复，就像朋友之间说话，不要像客服。

核心原则：
1. 情绪和感受优先。对方焦虑、烦躁、委屈、难过、无助时，先回应 ta 的心情，陪着 ta。很多时候对方只是想说说话，被听见就够了，不要急着给建议。
2. 建议只在对方明确求助或确实需要时给，而且要简洁。不要一听到延误、拥挤、末班车就自动进入"客服模式"推销解决方案。
3. 主动娱乐陪伴：当对方因为等车、延误而显得烦躁、无聊、不耐烦、难熬时，在自然共情之后，可以主动邀请玩一个轻松的小游戏来打发时间、缓解烦闷。每次只推荐一种，比如猜谜语、成语接龙、讲笑话、脑筋急转弯。邀请要自然简短，例如"等得挺烦的吧？要不咱们猜个谜语，我出题你来猜？"。
4. 区分情况：对方只是悲伤、想倾诉时，优先好好倾听，不要急着转移话题玩游戏；对方明显恐惧、愤怒或有安全需求时，先认真回应情绪和安全，绝不用玩笑敷衍；对方问实时列车信息时，诚实说查不到，绝不假装能查询。
5. 玩游戏时：猜谜语先出题不说答案，猜错给提示、猜对自然夸奖；成语接龙严格按规则——下一个成语的第一个字必须等于上一个成语的最后一个字，每次只让对方接一个明确的字，对方接错就指出并让其重试，不要乱接或跳字；讲笑话就直接讲一个短笑话。对方说"不玩了""没意思"或明显切换话题时就停下，回到正常聊天，不要反复邀请。
6. 对方说"饿了""困了"或只是想聊天时，按真实意图自然回应，不要强行安慰或强行拉着玩游戏。
7. 绝对不要用"我理解你""我能理解你的心情""我记得你"这类套话开头，也不要机械重复对方刚说的话。之前聊过的自然带一下就行。
8. 长度看情况，一两句话为主，最多 80 字。只输出回复正文，不要带前缀、括号或说明。"""

# 模块级单例与加载锁，避免重复加载模型
_model_lock = threading.Lock()
_gen_lock = threading.Lock()
_model = None
_tokenizer = None
_load_failed = False


def _load_model() -> bool:
    """惰性加载本地模型。成功返回 True，失败返回 False（仅尝试一次）。"""
    global _model, _tokenizer, _load_failed
    if _model is not None:
        return True
    if _load_failed:
        return False
    with _model_lock:
        if _model is not None:
            return True
        if _load_failed:
            return False
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            path = settings.llm_model_path
            logger.info("加载本地 LLM: %s", path)
            _tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            dtype = torch.float16 if settings.llm_device == "cuda" else torch.float32
            _model = AutoModelForCausalLM.from_pretrained(
                path,
                dtype=dtype,
                device_map=settings.llm_device if settings.llm_device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            if settings.llm_device == "cpu":
                _model = _model.to("cpu")
            _model.eval()
            logger.info("本地 LLM 加载完成")
            return True
        except Exception:
            logger.exception("本地 LLM 加载失败，将回退模板")
            _load_failed = True
            _model = None
            _tokenizer = None
            return False


def _build_messages(
    perception: PerceptionResult,
    decision: DecisionResult,
    history: list[PerceptionResult],
    ctx: ReplyContext,
    game_ctx: dict | None = None,
) -> list[dict]:
    """组装对话消息：系统提示 + 背景情境（system）+ 最近多轮 + 当前原话。

    情绪/事件等识别结果作为"背景"放在独立的 system 消息里，不混入乘客原话，
    避免模型把标签复述进回复。同时明确告诉模型：识别结果仅供参考，以乘客原话为准。

    ``game_ctx`` 为游戏状态锚点信息（如正在猜的谜语题目与答案），由调用方
    在生成前准备好，注入为 system 消息，保证多轮游戏中模型知道答案与当前进度。
    """
    messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]

    # 游戏状态（最高优先级的背景，直接影响本轮该怎么回）
    if game_ctx:
        messages.append({"role": "system", "content": game_ctx["instruction"]})

    # 背景信息（独立 system 消息，不影响对话语气）
    bg_parts: list[str] = []

    # 对话模式提示：帮助模型判断该安慰还是该闲聊
    text = perception.text or ""
    is_just_venting = any(k in text for k in ("说说话", "聊聊天", "陪陪我", "听我说", "吐槽", "发泄", "倾诉", "心里难受"))
    is_sad = any(k in text for k in ("难过", "伤心", "哭", "委屈", "想哭"))
    is_normal_chat = any(k in text for k in ("饿了", "笑话", "无聊", "困了", "累了想歇会", "听首歌", "讲个"))
    is_asking_info = any(k in text for k in ("多久", "什么时候", "查一下", "看看", "还有几站", "到哪了", "列车"))
    is_bored_restless = any(k in text for k in ("无聊", "烦死了", "不耐烦", "难熬", "等不下去", "太慢了", "急死人"))
    if game_ctx:
        # 游戏进行中：由 game_ctx 主导，不重复贴意图标签
        pass
    elif is_just_venting or is_sad:
        bg_parts.append("重要：乘客此刻情绪低落、只想有人陪着听 ta 说说话。请专注倾听和共情，绝对不要主动提议玩游戏、讲笑话或转移话题，也不要急着给建议。")
    elif is_normal_chat and not is_asking_info:
        bg_parts.append("这是普通闲聊或日常需求，正常回应即可，不要强行安慰。")
    elif is_asking_info:
        bg_parts.append("乘客在问实时信息，但你没有查询能力，诚实说明查不到即可，不要编造。")
    elif is_bored_restless:
        bg_parts.append("乘客等得有些烦躁无聊，共情后可以自然地邀请玩个小游戏打发时间。")

    if perception.emotion and perception.emotion != "平静":
        bg_parts.append(
            f"情绪模型初步判断乘客为「{perception.emotion}」（置信度{perception.emotion_score:.2f}）。"
            "注意：这只是参考，若与乘客实际表达不符，以乘客自己说的为准。"
        )
    if perception.event and perception.event != "其他/未知事件":
        bg_parts.append(f"当前情境：{perception.event}。")
    details: list[str] = []
    if ctx.urgency_target:
        src = "（前面提过）" if ctx.inherited_target else ""
        details.append(f"乘客要赶{ctx.urgency_target}{src}")
    if ctx.person_waiting:
        src = "（前面提过）" if ctx.inherited_person else ""
        details.append(f"{ctx.person_waiting}在等{src}")
    if ctx.last_train_gone:
        details.append("可能错过了末班车")
    if ctx.crowd_detail:
        details.append(f"车厢{ctx.crowd_detail}")
    if ctx.has_time_pressure:
        details.append("时间比较紧")
    if details:
        bg_parts.append("已知背景：" + "；".join(details) + "。")
    if bg_parts:
        messages.append({"role": "system", "content": "【仅供你判断参考，不要在回复里提到这些标签或分析】\n" + "\n".join(bg_parts)})

    # 最近几轮对话原文（乘客 → 助手），让模型自然承接
    for h in history[-4:]:
        if getattr(h, "text", None):
            messages.append({"role": "user", "content": h.text})
        prev_reply = h.extras.get("last_reply") if isinstance(h.extras, dict) else None
        if isinstance(prev_reply, str) and prev_reply:
            messages.append({"role": "assistant", "content": prev_reply})

    # 当前轮：只放乘客原话，让模型直接回应
    messages.append({"role": "user", "content": perception.text})
    return messages


def _generate(messages: list[dict]) -> Optional[str]:
    """实际执行生成（在锁内调用）。"""
    if _tokenizer is None or _model is None:
        return None
    try:
        import torch

        text = _tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = _tokenizer(text, return_tensors="pt").to(_model.device)
        with torch.no_grad():
            out = _model.generate(
                **inputs,
                max_new_tokens=settings.llm_max_new_tokens,
                temperature=settings.llm_temperature,
                do_sample=settings.llm_temperature > 0,
                top_p=0.9,
                repetition_penalty=1.15,
                pad_token_id=_tokenizer.eos_token_id,
            )
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        reply = _tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        return reply or None
    except Exception:
        logger.exception("LLM 生成异常")
        return None


def _cleanup(reply: str) -> str:
    """轻量后置清洗：剔除模型偶发的套话开头，让回复更自然。"""
    import re

    s = reply.strip()
    # 去掉以套话开头的短语（如"我理解你的焦急，"），保留后半句
    s = re.sub(r"^(我(?:能)?理解(?:你|您)?(?:的)?(?:心情|焦急|感受|着急|无奈|烦躁)?[，,。\s]*)", "", s)
    s = re.sub(r"^(您的心情我们理解[，,。\s]*)", "", s)
    s = re.sub(r"^(我(?:还)?记得(?:你)?(?:说)?(?:过)?[，,。\s]*)", "", s)
    s = re.sub(r"^(我知道你[，,。\s]*)", "", s)
    s = re.sub(r"^(您别急[，,。\s]*)", "", s)
    s = re.sub(r"^(先别着急[，,。\s]*)", "", s)
    return s.strip()


def _is_acceptable(reply: str, perception: PerceptionResult) -> bool:
    """轻量校验：过滤明显不合格的输出（空、含角色扮演残留、过长、编造迹象）。"""
    if not reply or len(reply) < 2:
        return False
    # 含系统提示泄露或角色扮演残留
    bad_markers = ("参考信息", "情绪模型", "置信度", "assistant", "role", "我的设定", "作为一个", "【仅供")
    if any(m in reply for m in bad_markers):
        return False
    # 禁止虚构查询能力：声称正在/已经执行不存在的查询操作
    fabrication_markers = (
        "我帮你查", "我帮您查", "我查一下", "我查查看", "我来查", "帮你查询", "帮您查询",
        "我已经确认", "我已经查了", "查到了", "查询结果", "实时查询", "我看了一下线路",
        "系统显示", "数据显示", "目前显示",
    )
    if any(m in reply for m in fabrication_markers):
        return False
    # 疑似编造具体线路/时间断言：出现"X号线""预计X分钟""还有X站到站"这类
    import re

    if re.search(r"\d+号线.*(到|来|还有|预计|停运|故障)", reply):
        return False
    if re.search(r"预计\d+分钟", reply):
        return False
    if len(reply) > 240:
        return False
    # 过滤中英混杂（模型偶发输出小写英文单词夹杂中文，如"wanna 先聊会儿天"；大写缩写如 APP 放行）
    if re.search(r"[a-z]{3,}", reply):
        return False
    return True


def _prepare_game_context(
    session_id: str | None,
    text: str,
    history: list[PerceptionResult],
) -> dict | None:
    """生成本轮回复前，先处理游戏状态，返回需要注入的 game_ctx（或 None）。

    覆盖三种情况：
    1. 用户明确拒绝/退出 → 结束游戏，返回 None（回到正常聊天）。
    2. 已有进行中的猜谜游戏 → 用户本轮输入视为一次猜答，判定对错并注入指引。
    3. 上一轮助手邀请了猜谜/成语接龙，用户本轮简短接受 → 开启对应游戏。
    """
    if not session_id:
        return None

    # 情况 1：拒绝或退出
    if games.is_rejection(text):
        if games.active_game(session_id):
            games.end_game(session_id)
            logger.info("[游戏] 用户退出，已清除游戏状态 | session=%s", session_id)
        return None

    last_reply = games.last_assistant_reply(history)
    invited_riddle = ("猜谜" in last_reply or "谜语" in last_reply or "谜面" in last_reply)
    invited_chain = ("成语接龙" in last_reply)
    explicit_riddle = ("猜谜" in text or "谜语" in text or "猜个" in text)
    explicit_chain = ("成语接龙" in text)
    has_active = bool(games.active_game(session_id))
    wants_more = any(k in text for k in ("再来", "继续", "还要")) and invited_riddle

    # 情况 2（最高优先）：用户明确要求开始/重开某游戏
    #   - 显式关键词（"成语接龙""猜谜"）：无论当前状态，直接重新开局
    #   - 简短接受邀请（"好啊""来吧"）：仅在当前没有进行中游戏时生效，
    #     避免把"来日方长"这类正常游戏输入误判为接受
    if explicit_riddle or wants_more or ((not has_active) and games.is_acceptance(text) and invited_riddle):
        games.end_game(session_id)
        r = games.start_riddle(session_id)
        if r:
            logger.info("[游戏] 开启猜谜 | session=%s q=%s", session_id, r.q)
            return {"direct_reply": games.compose_riddle_start(r)}
    if explicit_chain or ((not has_active) and games.is_acceptance(text) and invited_chain):
        games.end_game(session_id)
        init = games.start_chain(session_id)
        if init:
            logger.info("[游戏] 开启成语接龙 | session=%s idiom=%s", session_id, init["idiom"])
            return {"direct_reply": games.compose_chain_start(init["idiom"], init["required_char"])}

    # 情况 3：进行中的猜谜
    st = games.active_game(session_id)
    if st and st.get("game") == "riddle":
        res = games.submit_guess(session_id, text)
        if res["correct"]:
            instr = (
                f"当前在玩猜谜语。题目是：{st['q']}。乘客猜的是「{text}」，猜对了！"
                f"请自然地夸两句，然后问要不要再来一题。不要重复题目原文。"
            )
            logger.info("[游戏] 猜谜答对 | session=%s", session_id)
        else:
            hint_part = f"给个提示：{res['hint']}。" if res["hint"] else f"提示用完啦，答案其实是「{res['answer']}」。"
            instr = (
                f"当前在玩猜谜语。题目是：{st['q']}，答案是「{res['answer']}」（绝对不要直接告诉乘客答案）。"
                f"乘客猜的是「{text}」，猜错了。{hint_part}请自然地回应，语气轻松。"
            )
            logger.info("[游戏] 猜谜答错，给提示 | session=%s hint=%s", session_id, res["hint"])
        return {"instruction": instr}

    # 情况 4：进行中的成语接龙 —— 由 Python 校验并组合回复，保证规则与状态正确
    if st and st.get("game") == "chain":
        res = games.submit_chain(session_id, text)
        if res.get("topic_switch"):
            logger.info("[游戏] 接龙中切换话题，退出游戏 | session=%s", session_id)
            return None
        reply = games.compose_chain_reply(res)
        logger.info("[游戏] 接龙回合 valid=%s | session=%s", res["valid"], session_id)
        return {"direct_reply": reply}

    return None


def generate_reply(
    perception: PerceptionResult,
    decision: DecisionResult,
    history: list[PerceptionResult] | None = None,
    session_id: str | None = None,
) -> Optional[str]:
    """尝试用本地 LLM 生成一条回复。

    返回：合格的回复字符串；任何不可用情况返回 ``None``（调用方应回退模板）。
    超时由 ``settings.llm_timeout`` 控制。所有分支均有明确日志。
    """
    # 分支 1：配置禁用
    if not settings.llm_enabled:
        logger.info("[LLM] 已禁用 (llm_enabled=False)，跳过生成本地大模型回复")
        return None

    # 分支 2：模型加载失败（只尝试一次，后续直接跳过）
    if not _load_model():
        logger.info("[LLM] 模型不可用，跳过生成（回退模板）")
        return None

    history = history or []
    ctx = extract_context(
        perception.text,
        history_texts=[h.text for h in history if getattr(h, "text", None)],
    )
    game_ctx = _prepare_game_context(session_id, perception.text or "", history)

    # 游戏直出回复（猜谜出题、成语接龙等）：使用 Python 组合好的可靠回复，跳过模型生成
    if game_ctx and "direct_reply" in game_ctx:
        reply = game_ctx["direct_reply"]
        logger.info("[LLM] 游戏直出回复 (跳过生成) | source=llm_local len=%d", len(reply))
        return reply

    messages = _build_messages(perception, decision, history, ctx, game_ctx=game_ctx)
    logger.info("[LLM] 开始生成：history=%d 轮, 策略=%s, game=%s", len(history), decision.strategy, "on" if game_ctx else "off")

    holder: dict[str, Optional[str]] = {"reply": None, "error": None}

    def _worker():
        try:
            with _gen_lock:
                holder["reply"] = _generate(messages)
        except Exception as e:  # 捕获线程内未预期异常，避免吞掉
            holder["error"] = repr(e)

    t = threading.Thread(target=_worker, daemon=True)
    start = time.monotonic()
    t.start()
    t.join(timeout=settings.llm_timeout)
    elapsed = time.monotonic() - start

    # 分支 3：超时
    if t.is_alive():
        logger.warning(
            "[LLM] 生成超时 (%.1fs > %.1fs 限制)，回退模板 | source=template_fallback reason=timeout",
            elapsed, settings.llm_timeout,
        )
        return None

    # 分支 4：线程内异常
    if holder["error"]:
        logger.error(
            "[LLM] 生成异常 (%.1fs)，回退模板 | source=template_fallback reason=exception detail=%s",
            elapsed, holder["error"],
        )
        return None

    reply = holder["reply"]

    # 轻量清洗套话开头（不改变是否合格的判定，只让回复更自然）
    if reply:
        cleaned = _cleanup(reply)
        if cleaned != reply:
            logger.info("[LLM] 清洗套话开头: %r -> %r", reply[:30], cleaned[:30])
            reply = cleaned

    # 分支 5：生成返回空
    if not reply:
        logger.info(
            "[LLM] 生成返回空 (%.1fs)，回退模板 | source=template_fallback reason=empty_output",
            elapsed,
        )
        return None

    # 分支 6：输出不合格
    if not _is_acceptable(reply, perception):
        logger.info(
            "[LLM] 输出校验未通过 (%.1fs len=%d) 内容=%r，回退模板 | source=template_fallback reason=rejected",
            elapsed, len(reply), reply[:80],
        )
        return None

    # 分支 7：成功
    logger.info(
        "[LLM] 生成成功 (%.1fs len=%d) | source=llm_local",
        elapsed, len(reply),
    )
    return reply


def reset_for_tests() -> None:
    """测试用：清空已加载模型、失败标记与游戏状态。"""
    global _model, _tokenizer, _load_failed
    _model = None
    _tokenizer = None
    _load_failed = False
    games._states.clear()
