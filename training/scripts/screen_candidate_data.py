# -*- coding: utf-8 -*-
"""候选公开数据导入与筛选执行脚本。

读取 data/candidate/manifest.json，对其中 allowed_to_screen=true 且文件存在的
候选数据集执行 screening，并将结果写入 data/candidate/screened/：
  - candidate_screening.csv   逐条结果（来源/原始标签/映射标签/状态/原因/模型辅助）
  - candidate_report.json     类别分布与质量统计
  - candidate_review.csv      仅「待复核」子集，供人工审阅
  - candidate_keep.csv        仅「保留」子集（仍未合并进正式训练集）

不修改任何正式数据集；不训练模型。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))

from data_tools.screening import (  # noqa: E402
    KEEP, REVIEW, build_report, load_existing_texts, results_to_frame, screen_dataframe,
)

CAND_DIR = ROOT / "data" / "candidate"
RAW_DIR = CAND_DIR / "raw"
OUT_DIR = CAND_DIR / "screened"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((CAND_DIR / "manifest.json").read_text(encoding="utf-8"))

    existing = load_existing_texts()
    all_results = []

    for ds in manifest.get("datasets", []):
        if not ds.get("allowed_to_screen"):
            print(f"[跳过] {ds['id']}: 未获筛选许可 (license_verified={ds.get('license_verified')})")
            continue
        fname = ds.get("file")
        if not fname:
            print(f"[跳过] {ds['id']}: 未提供本地文件")
            continue
        fp = RAW_DIR / fname
        if not fp.exists():
            print(f"[跳过] {ds['id']}: 文件不存在 {fp}")
            continue
        df = pd.read_csv(fp) if fp.suffix.lower() == ".csv" else pd.read_excel(fp)
        print(f"[处理] {ds['id']} rows={len(df)}")
        results = screen_dataframe(
            df,
            source=ds["id"],
            text_col=ds.get("text_col", "text"),
            label_col=ds.get("label_col", "label"),
            existing=existing,
            # 本轮不启用模型辅助（避免长时间加载；测试中已覆盖该分支）
            model_predict=None,
        )
        all_results.extend(results)

    frame = results_to_frame(all_results)
    report = build_report(all_results)

    frame.to_csv(OUT_DIR / "candidate_screening.csv", index=False, encoding="utf-8-sig")
    frame[frame["status"] == KEEP].to_csv(OUT_DIR / "candidate_keep.csv", index=False, encoding="utf-8-sig")
    frame[frame["status"] == REVIEW].to_csv(OUT_DIR / "candidate_review.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "candidate_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n结果已写入: {OUT_DIR}")


if __name__ == "__main__":
    main()
