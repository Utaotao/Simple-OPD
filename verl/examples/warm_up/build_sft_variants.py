"""
Build two SFT-ready parquets from the paired outputs of
build_pairs_from_rollout.py + GPT-5.5 rewrite:

  (a) sft_pos_no_cot.parquet
      Take pairs/sft_pos.parquet and strip the chain-of-thought from `response`,
      keeping only the final "Answer: $X" line (using ground_truth as the
      answer, so this is a clean "answer-only SFT" dataset).

  (b) sft_pos_gpt55_ready.parquet
      Take derived/sft_pos_gpt55.parquet and swap the (verbose Qwen) `response`
      for `new_response` (GPT-5.5 rewrite, still contains reasoning but shorter
      and cleaner). Only keep rows where new_is_correct == True (the rewrite
      still lands on the correct answer). is_correct is refreshed accordingly
      so that filter_correct_sft.py --only_correct 1 keeps every row.

Both outputs preserve columns: prompt, response, is_correct, plus provenance
(pair_id, ground_truth, pred, data_source, extra_info, orig_index,
sample_index, score). This is the schema that
verl/examples/warm_up/filter_correct_sft.py expects.

Usage:
    python build_sft_variants.py \
        --pos_parquet  /abs/path/pairs/sft_pos.parquet \
        --gpt55_parquet /abs/path/derived/sft_pos_gpt55.parquet \
        --out_dir      /abs/path/derived
"""

import argparse
import os
import re

import pandas as pd


ANSWER_TEMPLATE = "Answer: ${answer}$"


def _strip_cot(row) -> str:
    """Return an answer-only response for one row.

    We prefer the model's own last "Answer: ..." line if present (some rows
    end with `Answer: $-3$` or `Answer: -3`), because that is exactly the
    format the eval regex expects.  If not found, synthesize one from
    ground_truth.
    """
    resp = str(row["response"])
    # Grab the last non-empty "Answer:" line, if any.
    matches = list(re.finditer(r"Answer:\s*[^\n]+", resp))
    if matches:
        return matches[-1].group(0).strip()
    gt = str(row.get("ground_truth", "")).strip()
    return f"Answer: ${gt}$"


def build_no_cot(pos_parquet: str, out_path: str) -> None:
    print(f"[no_cot] loading {pos_parquet}")
    df = pd.read_parquet(pos_parquet)
    print(f"[no_cot] input rows = {len(df)}, cols = {list(df.columns)}")
    if "response" not in df.columns:
        raise ValueError("sft_pos.parquet is expected to have a `response` column.")
    df = df.copy()
    df["response"] = df.apply(_strip_cot, axis=1)
    # is_correct is already True across sft_pos.parquet; keep as-is.
    if "is_correct" in df.columns:
        df["is_correct"] = df["is_correct"].astype(bool)
    print(f"[no_cot] sample response[0] = {df['response'].iloc[0]!r}")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[no_cot] wrote {len(df)} rows -> {out_path}")


def build_gpt55(gpt55_parquet: str, out_path: str) -> None:
    print(f"[gpt55] loading {gpt55_parquet}")
    df = pd.read_parquet(gpt55_parquet)
    print(f"[gpt55] input rows = {len(df)}, cols = {list(df.columns)}")
    for col in ("new_response", "new_is_correct"):
        if col not in df.columns:
            raise ValueError(f"gpt55 parquet must have column `{col}`.")

    # Keep only rows whose GPT-5.5 rewrite is still correct.
    before = len(df)
    keep = df["new_is_correct"].astype(bool)
    df = df[keep].reset_index(drop=True)
    print(f"[gpt55] kept new_is_correct=True: {len(df)}/{before}")

    # Swap the response for the GPT-5.5 rewrite; refresh is_correct/pred/score
    # so downstream filters see a consistent "positive-only" dataset.
    df["response"] = df["new_response"].astype(str)
    df["is_correct"] = True
    if "new_pred" in df.columns:
        df["pred"] = df["new_pred"]
    if "new_score" in df.columns:
        df["score"] = df["new_score"]

    # Drop the now-redundant `new_*` columns to keep the schema clean.
    drop_cols = [c for c in ("new_response", "new_pred", "new_is_correct",
                             "new_score", "error") if c in df.columns]
    df = df.drop(columns=drop_cols)
    print(f"[gpt55] sample response[0] head = {df['response'].iloc[0][:200]!r}...")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[gpt55] wrote {len(df)} rows -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos_parquet", required=True,
                    help="pairs/sft_pos.parquet (verbose Qwen CoT responses).")
    ap.add_argument("--gpt55_parquet", required=True,
                    help="derived/sft_pos_gpt55.parquet (GPT-5.5 rewritten "
                         "responses; must have new_response/new_is_correct).")
    ap.add_argument("--out_dir", required=True,
                    help="directory to write sft_pos_no_cot.parquet and "
                         "sft_pos_gpt55_ready.parquet into.")
    ap.add_argument("--no_cot_name", default="sft_pos_no_cot.parquet")
    ap.add_argument("--gpt55_name",  default="sft_pos_gpt55_ready.parquet")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    build_no_cot(args.pos_parquet, os.path.join(args.out_dir, args.no_cot_name))
    build_gpt55(args.gpt55_parquet, os.path.join(args.out_dir, args.gpt55_name))


if __name__ == "__main__":
    main()
