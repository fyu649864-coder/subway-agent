"""回复策略调整回归测试：真实调用本地 LLM，验证安慰优先与无虚构能力。"""
import json
import time
import urllib.request

BASE = "http://127.0.0.1:8000/api/v1/chat"

SCENARIOS = [
    "地铁怎么还不来，我马上就要考试了！",
    "今天真的太倒霉了，什么事情都不顺。",
    "挤成这样，我真的烦透了。",
    "我只是想找个人说说话。",
    "我饿了。",
    "给我讲个笑话。",
    "帮我看看地铁还要多久到。",
]

# 连续对话：焦虑 → 逐渐转为普通聊天
CONVO = [
    "明天要考试，我现在好焦虑啊。",
    "现在地铁又延误了，真的烦死了。",
    "算了不说了，你会讲笑话吗？",
]


def chat(session_id: str, text: str) -> dict:
    payload = json.dumps({"session_id": session_id, "text": text}).encode("utf-8")
    req = urllib.request.Request(
        BASE, data=payload, headers={"Content-Type": "application/json"}
    )
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=60) as resp:
        elapsed = time.monotonic() - start
        data = json.loads(resp.read().decode("utf-8"))
    return data, elapsed


def main():
    print("=" * 70)
    print("单场景测试（真实本地模型）")
    print("=" * 70)
    for i, text in enumerate(SCENARIOS, 1):
        sid = f"test_strategy_{i}"
        data, elapsed = chat(sid, text)
        reply = data.get("reply", "")
        source = data.get("extras", {}).get("reply_source", "?")
        print(f"\n[{i}] 用户: {text}")
        print(f"    回复({source}, {elapsed:.1f}s): {reply}")
        # 检查
        bad = [m for m in ("我帮你查", "我查一下", "我已经确认", "我理解你", "我记得你") if m in reply]
        if bad:
            print(f"    ⚠️  命中禁用短语: {bad}")
        else:
            print(f"    ✅ 无禁用短语")

    print("\n" + "=" * 70)
    print("连续对话测试（焦虑 → 普通聊天）")
    print("=" * 70)
    sid = "test_convo_strategy"
    for i, text in enumerate(CONVO, 1):
        data, elapsed = chat(sid, text)
        reply = data.get("reply", "")
        source = data.get("extras", {}).get("reply_source", "?")
        print(f"\n[轮{i}] 用户: {text}")
        print(f"      回复({source}, {elapsed:.1f}s): {reply}")
        bad = [m for m in ("我帮你查", "我查一下", "我已经确认", "我理解你", "我记得你", "我还记得") if m in reply]
        if bad:
            print(f"      ⚠️  命中禁用短语: {bad}")
        else:
            print(f"      ✅ 无禁用短语")


if __name__ == "__main__":
    main()
