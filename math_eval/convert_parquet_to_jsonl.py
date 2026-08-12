"""Convert verl-format math benchmark parquet -> jsonl for eval_math.py.

Handles all 5 datasets we want to evaluate:
  AMC23, AIME2024, AIME2025, HMMT2024, HMMT2025

Each output line: {"problem": <clean question>, "answer": <ground_truth>}

Notes
-----
- The `extra_info.question` field already strips the training-time prompt
  suffix ("Let's think step by step and output the final answer within
  \\boxed{}."), which is what we want, because eval_math.py adds its own
  suffix automatically.
- AIME2024's ground_truth is wrapped as "\\boxed{204}"; we strip the
  wrapper to keep all datasets consistent (math_verify can parse either,
  but consistency is nicer).
"""
import json
import os
import re
import pandas as pd

GOPD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.environ.get(
    "DATA_ROOT", os.path.join(os.path.dirname(GOPD_ROOT), "G-OPD-Training-Data")
)
DST_ROOT = os.path.join(GOPD_ROOT, "data")

# (parquet sub-folder, jsonl sub-folder)
DATASETS = [
    ("AMC23",    "amc23"),
    ("AIME2024", "aime24"),
    ("AIME2025", "aime25"),
    ("HMMT2024", "hmmt24"),
    ("HMMT2025", "hmmt25"),
]

BOXED_RE = re.compile(r"^\s*\\boxed\{(.*)\}\s*$", re.DOTALL)


def extract_question(row):
    extra = row.get("extra_info")
    if isinstance(extra, dict) and extra.get("question"):
        return extra["question"]
    prompt = row.get("prompt")
    if prompt is not None and len(prompt) > 0:
        msg = prompt[0]
        if isinstance(msg, dict):
            return msg.get("content", "")
    return ""


def extract_answer(row):
    rm = row.get("reward_model")
    if not isinstance(rm, dict):
        return ""
    gt = rm.get("ground_truth")
    if gt is None:
        return ""
    gt = str(gt).strip()
    m = BOXED_RE.match(gt)
    if m:
        gt = m.group(1).strip()
    return gt


def convert(src_dir, dst_dir):
    src = os.path.join(SRC_ROOT, src_dir, "test.parquet")
    dst = os.path.join(DST_ROOT, dst_dir, "test.jsonl")
    if not os.path.exists(src):
        print(f"[skip] {src_dir}: not found -> {src}")
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    df = pd.read_parquet(src)
    n = 0
    with open(dst, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            problem = extract_question(row)
            answer = extract_answer(row)
            if not problem or not answer:
                continue
            f.write(json.dumps({"problem": problem, "answer": answer}, ensure_ascii=False) + "\n")
            n += 1
    print(f"[done] {src_dir:>10s} -> {dst}  ({n} samples)")


def main():
    for src_dir, dst_dir in DATASETS:
        convert(src_dir, dst_dir)


if __name__ == "__main__":
    main()
