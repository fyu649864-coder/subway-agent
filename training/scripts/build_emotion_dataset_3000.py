# -*- coding: utf-8 -*-
"""
构建 7 类情绪分类数据集（约 3000 条）。
- 仅以 augmented_metro.xlsx 中 112 条原始样本为种子，不复用原有 6067 条增强。
- 57 类原始标签 → 7 类（平静/焦虑/愤怒/悲伤/恐惧/烦躁/无助），按语义归并。
- 按原始 group（序号）划分 train/val/test，同组绝不跨集合。
- val/test 仅保留真实原始文本，不做增强。
- 仅对 train 做模板化受控增强，保持语义与情绪一致，覆盖多种地铁场景。
- 输出 xlsx + label_mapping.json + 质量检查；另写报告由外部生成。
不训练、不下载、不改原始 xlsx、不动 models/。
"""
import json
import random
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

RAW_XLSX = r"C:\Users\64513\Desktop\augmented_metro.xlsx"
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 20261009
random.seed(SEED)
np.random.seed(SEED)

# ───────────────────────── 57 → 7 情绪归并映射 ─────────────────────────
LABEL_MAP_57_TO_7 = {
    # 平静（含中性、低唤醒正面：满意/认可/轻松/礼貌/理解/歉意/建议等）
    "平静": "平静", "平和": "平静", "平复": "平静", "克制": "平静", "轻松": "平静",
    "礼貌": "平静", "歉意": "平静", "理解": "平静", "建议": "平静", "满意": "平静",
    "舒心": "平静", "认可": "平静", "开心": "平静", "愉悦": "平静", "庆幸": "平静",
    "感激": "平静", "感动": "平静", "暖心": "平静", "敬佩": "平静", "欣慰": "平静",
    "善意": "平静", "激动": "平静",
    # 焦虑（对未来/未知的担忧与困惑）
    "焦虑": "焦虑", "焦急": "焦虑", "着急": "焦虑", "担心": "焦虑", "担忧": "焦虑",
    "紧张": "焦虑", "疑惑": "焦虑", "困惑": "焦虑", "窘迫": "焦虑",
    # 愤怒（对人/事的气愤与不满）
    "愤怒": "愤怒", "生气": "愤怒", "恼火": "愤怒", "不满": "愤怒", "反感": "愤怒",
    "厌恶": "愤怒", "厌烦": "愤怒", "嘲讽": "愤怒", "抱怨": "愤怒", "吐槽": "愤怒",
    # 悲伤（失望、沮丧、委屈、懊悔）
    "失望": "悲伤", "沮丧": "悲伤", "委屈": "悲伤", "懊悔": "悲伤", "懊恼": "悲伤",
    # 恐惧（恐慌、惊吓）
    "恐慌": "恐惧", "惊吓": "恐惧",
    # 烦躁（烦躁、身体不适、无语）
    "烦躁": "烦躁", "难受": "烦躁", "不适": "烦躁", "疲惫": "烦躁",
    "反胃": "烦躁", "无语": "烦躁",
    # 无助（无奈、压抑、不便）
    "无奈": "无助", "压抑": "无助", "不便": "无助",
}
CLASS_7 = ["平静", "焦虑", "愤怒", "悲伤", "恐惧", "烦躁", "无助"]

# ───────────────────────── 读取与清洗 ─────────────────────────
df = pd.read_excel(RAW_XLSX)
orig = df[df["序号"].notna()].copy().reset_index(drop=True)
orig["序号"] = orig["序号"].astype(int)
orig["乘客说的话"] = orig["乘客说的话"].astype(str).str.strip()
orig = orig[(orig["乘客说的话"] != "") & orig["情绪"].notna() &
            (orig["乘客说的话"].str.len() >= 4)]
orig = orig.drop_duplicates(subset=["乘客说的话"]).reset_index(drop=True)
orig["emotion"] = orig["情绪"].map(LABEL_MAP_57_TO_7)
assert orig["emotion"].notna().all(), f"未映射标签: {orig[orig['emotion'].isna()]['情绪'].unique()}"

print(f"[INFO] 有效原始样本: {len(orig)} 条；7 类分布: {dict(Counter(orig['emotion']))}")

# ───────────────────────── 按组划分 train/val/test ─────────────────────────
# 各类 test/val/train 的原始 group 数量
SPLIT_PLAN = {
    "平静": {"test": 6, "val": 6, "train": 31},
    "愤怒": {"test": 3, "val": 3, "train": 16},
    "焦虑": {"test": 3, "val": 3, "train": 11},
    "烦躁": {"test": 2, "val": 2, "train": 11},
    "悲伤": {"test": 1, "val": 1, "train": 5},
    "无助": {"test": 1, "val": 1, "train": 3},
    "恐惧": {"test": 1, "val": 0, "train": 2},
}
rng = np.random.RandomState(SEED)
orig["split"] = ""
for emo, plan in SPLIT_PLAN.items():
    gids = orig[orig["emotion"] == emo]["序号"].tolist()
    rng.shuffle(gids)
    assert len(gids) == plan["test"] + plan["val"] + plan["train"], (emo, len(gids), plan)
    t, v = plan["test"], plan["test"] + plan["val"]
    orig.loc[orig["序号"].isin(gids[:t]), "split"] = "test"
    orig.loc[orig["序号"].isin(gids[t:v]), "split"] = "validation"
    orig.loc[orig["序号"].isin(gids[v:]), "split"] = "train"

rows = []
for _, r in orig.iterrows():
    rows.append({
        "text": r["乘客说的话"], "emotion": r["emotion"],
        "group_id": int(r["序号"]), "source": "original",
        "is_augmented": 0, "split": r["split"], "scene": r["场景"],
    })

# ───────────────────────── 模板化增强（仅 train） ─────────────────────────
# 场景 → 可插入的情境短语（丰富覆盖用户列出的场景）
SCENE_CTX = {
    "列车延误": ["赶飞机", "赶高铁", "赶考试", "要面试", "上班打卡", "接孩子放学",
                "换乘末班车", "约了客户", "赶着去医院"],
    "列车急刹": ["扶着扶手", "抱着孩子", "拎着早餐", "低头看手机", "站在门口"],
    "列车拥挤": ["早高峰", "晚高峰", "下雨天人更多", "节假日", "赶着换乘",
                "背着大包", "推着婴儿车"],
    "安检排队": ["赶着上班", "拎着水", "带着充电宝", "排队二十分钟", "着急赶车"],
    "宠物气味": ["带着早餐", "坐着犯困", "空气不流通", "车厢封闭"],
    "导向不清": ["第一次坐这线", "赶时间", "拎着行李", "看不懂箭头"],
    "抢座占座": ["带老人出行", "抱着小孩", "站了一路", "拎着大件行李"],
    "报站错误": ["第一次来这座城市", "要换乘", "快到站了", "戴着耳机"],
    "服务问题": ["问了半天没人理", "赶时间", "人生地不熟", "带着老人"],
    "正常乘车": ["今天运气好", "难得人少", "下班路上", "不用挤"],
    "物品掉落轨道": ["手机掉了", "钱包掉了", "资料很重要", "差点冲动下去捡"],
    "特殊乘客": ["推着轮椅", "带着视障家人", "孕妇出行", "老人独自乘车"],
    "空调温度异常": ["穿得少", "刚跑完步", "抱着睡着的孩子", "满头大汗"],
    "节假日大客流": ["国庆出游", "春节返程", "演唱会散场", "烟花表演结束"],
    "行李不便": ["拖着两个箱子", "背着电脑包", "拎着特产", "推着婴儿车"],
    "设备故障": ["屏蔽门", "扶梯", "售票机", "电梯", "显示屏"],
    "车厢噪音": ["外放视频", "大声打电话", "孩子哭闹", "聊天特别大声"],
    "车厢异味": ["早餐味", "烟味", "榴莲味", "香水太浓", "汗味"],
    "闸机故障": ["刷手机", "刷交通卡", "扫二维码", "卡被吞了"],
    "首末班车": ["赶末班车", "没赶上末班车", "末班延运", "就差一步"],
}
GENERIC_CTX = ["今天", "早上", "晚高峰", "又", "这会儿", "刚才", "每次"]

INTENSIFIERS = ["太", "真", "特别", "非常", "超级", "简直", "可", "实在"]
PARTICLES = ["了", "啦", "啊", "呀", "呢", "吧", "呗", "嘛"]
EMO_WORDS = {
    "焦虑": ["着急", "揪心", "心里没底", "慌慌的", "七上八下", "不踏实", "悬着", "发慌"],
    "愤怒": ["气死我了", "火大", "来气", "不爽", "窝火", "生气", "恼火", "忍不了"],
    "悲伤": ["难受", "心寒", "失望透顶", "委屈", "不是滋味", "沮丧", "难过", "心凉"],
    "恐惧": ["吓人", "后怕", "心里发毛", "紧张", "慌了", "吓一跳", "惊魂未定", "害怕"],
    "烦躁": ["烦", "躁得慌", "闷", "受不了", "头疼", "膈应", "闹心", "糟心"],
    "无助": ["没办法", "没辙", "只能忍着", "叫天天不应", "孤立无援", "干着急", "无可奈何", "无助"],
    "平静": ["还好", "没事", "习惯了", "无所谓", "不急", "放平心态", "平常心", "可以接受"],
}

# 句式模板：{c}=原句核心短语, {x}=情境, {i}=程度词, {e}=情绪词, {p}=语气词
TEMPLATES = [
    "{c}，{i}{e}{p}",
    "碰上{x}，{c}，{i}{e}{p}",
    "{c}，真让人{e}{p}",
    "又遇到{c}，{i}{e}",
    "{x}的时候{c}，{i}{e}{p}",
    "{i}{e}，{c}{p}",
    "每次都这样：{c}，{i}{e}",
    "{c}就算了，{x}还这样，{i}{e}{p}",
    "一大早的{c}，{i}{e}",
    "{c}？{i}{e}了都",
    "不是我说，{c}，{i}{e}{p}",
    "{x}，{c}，心情{e}",
    "{e}{p}，{c}",
    "能不能别这样了，{c}，{i}{e}",
    "{c}，换谁都{e}吧",
    "好家伙，{c}，{i}{e}{p}",
    "{x}本来就急，{c}，{i}{e}",
    "{c}，{i}{e}，有没有人管管",
    "唉，{c}，{e}",
    "{c}，{e}得不行",
]


def extract_core(text: str) -> str:
    """取原句核心：去掉末尾轻语气词，截取关键片段。"""
    s = text.rstrip("。！？?!，, ")
    # 若含逗号，取后半句（往往是情绪/评价核心）
    if "，" in s:
        parts = [p for p in s.split("，") if len(p) >= 3]
        if parts:
            s = parts[-1]
    return s[:30]


def gen_variants(orig_text: str, scene: str, emo: str, n: int,
                 existing: set) -> list:
    core = extract_core(orig_text)
    ctx_pool = SCENE_CTX.get(scene, []) + GENERIC_CTX
    e_words = EMO_WORDS[emo]
    out, tries = [], 0
    while len(out) < n and tries < n * 60:
        tries += 1
        tpl = random.choice(TEMPLATES)
        s = tpl.format(
            c=core,
            x=random.choice(ctx_pool),
            i=random.choice(INTENSIFIERS),
            e=random.choice(e_words),
            p=random.choice(PARTICLES),
        )
        s = s.replace("，，", "，").replace("。。", "。").strip("， ")
        if len(s) < 8 or s in existing:
            continue
        # 避免与原句过像（简单换词伪增强）
        if SequenceMatcher(None, s, orig_text).ratio() > 0.78:
            continue
        # 避免与已生成句过像
        if any(SequenceMatcher(None, s, o).ratio() > 0.88 for o in out[-30:]):
            continue
        existing.add(s)
        out.append(s)
    return out


# 每类 train 目标总量
TRAIN_TARGET = {"平静": 511, "愤怒": 490, "焦虑": 480, "烦躁": 480,
                "悲伤": 400, "无助": 350, "恐惧": 250}

train_orig = orig[orig["split"] == "train"].copy()
global_texts = {r["text"] for r in rows}
for emo in CLASS_7:
    sub = train_orig[train_orig["emotion"] == emo].sort_values("序号").reset_index(drop=True)
    n_orig = len(sub)
    need = TRAIN_TARGET[emo] - n_orig
    if need <= 0:
        continue
    base = need // n_orig
    extra = need % n_orig
    for i, (_, r) in enumerate(sub.iterrows()):
        k = base + (1 if i < extra else 0)
        variants = gen_variants(r["乘客说的话"], r["场景"], emo, k, global_texts)
        for v in variants:
            rows.append({
                "text": v, "emotion": emo, "group_id": int(r["序号"]),
                "source": "augmentation", "is_augmented": 1, "split": "train",
                "scene": r["场景"],
            })
    got = sum(1 for x in rows if x["emotion"] == emo and x["split"] == "train")
    print(f"[AUG] {emo}: 目标 {TRAIN_TARGET[emo]}，实际 {got}")

# ───────────────────────── 组装与质量检查 ─────────────────────────
out = pd.DataFrame(rows)
out = out.drop_duplicates(subset=["text"]).reset_index(drop=True)

# 泄漏检查：同一 group_id 不应跨 split
leak = out.groupby("group_id")["split"].nunique()
n_leak = int((leak > 1).sum())
# 空文本 / 异常标签
n_empty = int((out["text"].astype(str).str.strip() == "").sum())
bad_labels = [e for e in out["emotion"].unique() if e not in CLASS_7]

report = {
    "total": int(len(out)),
    "by_emotion": dict(Counter(out["emotion"])),
    "by_split": dict(Counter(out["split"])),
    "split_x_emotion": out.groupby(["split", "emotion"]).size().unstack(fill_value=0).to_dict(),
    "augmented_count": int(out["is_augmented"].sum()),
    "original_count": int((out["is_augmented"] == 0).sum()),
    "groups_cross_split": n_leak,
    "empty_texts": n_empty,
    "bad_labels": bad_labels,
    "num_groups": int(out["group_id"].nunique()),
}

out_path = OUT_DIR / "metro_emotion_dataset_3000.xlsx"
out.to_excel(out_path, index=False)
with open(OUT_DIR / "emotion_label_mapping.json", "w", encoding="utf-8") as f:
    json.dump({
        "label_57_to_7": LABEL_MAP_57_TO_7,
        "classes_7": CLASS_7,
        "split_plan_original_groups": SPLIT_PLAN,
        "train_target_per_class": TRAIN_TARGET,
        "quality_report": report,
    }, f, ensure_ascii=False, indent=2)

print("\n===== 质量报告 =====")
print(json.dumps(report, ensure_ascii=False, indent=2))
print(f"[OK] 已保存: {out_path}")
