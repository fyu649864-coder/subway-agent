"""场景评估脚本：对比模板回复与本地 LLM 回复的自然度、连贯性与响应时间。

运行方式（在 backend 目录下）：
    python ../scripts/eval_reply_quality.py

覆盖场景：赶考试、列车延误、车厢拥挤、末班车、家人等待、
无助被误判为愤怒、多轮连续对话。
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models import UserInput
from app.services.conversation_service import ConversationService

SCENARIOS: list[tuple[str, list[str]]] = [
    ("赶考试", ["马上要考试了，地铁怎么还不来，我要迟到了！"]),
    ("列车延误", ["这车晚点了吧，等了快二十分钟了，到底怎么回事"]),
    ("车厢拥挤", ["连续三班车都挤不上去，人也太多了"]),
    ("末班车", ["末班车已经走了，我回不去了，怎么办啊"]),
    ("家人等待", ["老婆还在医院等我，这车一直不来，急死了"]),
    ("无助被误判愤怒", ["我真的没办法了，感觉好无助，谁能帮帮我"]),
    (
        "多轮连续对话",
        [
            "我要赶去高铁站，车怎么还不来",
            "已经等了快半小时了",
            "再不来我真的要误车了",
        ],
    ),
]


def run_mode(use_llm: bool) -> list[dict]:
    from unittest.mock import patch

    from app.core.config import settings
    from app.response import llm_generator

    llm_generator.reset_for_tests()
    svc = ConversationService()

    # 基线模式：直接将生成器打桩为 None，强制走模板，且不加载模型。
    patch_ctx = patch(
        "app.services.response_service.llm_generator.generate_reply",
        return_value=None,
    )
    if not use_llm:
        patch_ctx.start()
    try:
        rows = []
        for name, turns in SCENARIOS:
            sid = f"eval_{name}_{int(use_llm)}"
            for i, text in enumerate(turns):
                t0 = time.monotonic()
                res = svc.handle(UserInput(text=text, session_id=sid))
                dt = time.monotonic() - t0
                rows.append(
                    {
                        "scenario": name,
                        "turn": i,
                        "llm": use_llm,
                        "reply": res.reply,
                        "source": res.template_id,
                        "llm_used": res.extras.get("llm_used", False),
                        "seconds": round(dt, 2),
                        "emotion": res.emotion,
                        "event": res.event,
                    }
                )
    finally:
        if not use_llm:
            patch_ctx.stop()
    return rows


def main():
    print("=" * 70)
    print("【模板模式】（LLM 关闭，作为接入前基线）")
    print("=" * 70)
    baseline = run_mode(False)
    for r in baseline:
        print(f"\n[{r['scenario']}#{r['turn']}] {r['emotion']}/{r['event']} "
              f"({r['seconds']}s)\n  > {r['reply']}")

    print("\n" + "=" * 70)
    print("【本地 LLM 模式】（Qwen2.5-3B-Instruct）")
    print("=" * 70)
    llm_rows = run_mode(True)
    for r in llm_rows:
        flag = "LLM" if r["llm_used"] else "FALLBACK"
        print(f"\n[{r['scenario']}#{r['turn']}] {r['emotion']}/{r['event']} "
              f"({r['seconds']}s, {flag})\n  > {r['reply']}")

    # 汇总
    print("\n" + "=" * 70)
    n_llm = sum(1 for r in llm_rows if r["llm_used"])
    avg_t = sum(r["seconds"] for r in llm_rows) / len(llm_rows)
    print(f"LLM 实际使用率: {n_llm}/{len(llm_rows)}")
    print(f"LLM 模式平均响应时间: {avg_t:.2f}s")
    base_avg = sum(r["seconds"] for r in baseline) / len(baseline)
    print(f"模板模式平均响应时间: {base_avg:.2f}s")


if __name__ == "__main__":
    main()
