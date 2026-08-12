"""
Filter a rollout parquet (produced by simple_infer_dapo17k.py / build_sft_from_generation.py)
down to the samples we actually want to SFT on.

Input parquet is expected to contain at least:
    - `prompt`      : str | list[{role,content}]
    - `response`    : str
    - `is_correct`  : bool     # True -> the model's rollout matches ground truth
    - `pair_id`     : int      # (optional) present when input is already a
                                 paired-SFT parquet from
                                 examples/g_opd/build_pairs_from_rollout.py

Filtering rule:
    - default: keep only rows with is_correct == True (SFT-on-correct-only).
    - --only_wrong 1: keep only rows with is_correct == False (SFT-on-wrong-only).
    - --only_correct 0 and --only_wrong 0: keep all rows.
    (--only_correct and --only_wrong cannot both be 1.)

Paired-SFT input (new format, produced by build_pairs_from_rollout.py):
    - Files like `sft_pos.parquet` / `sft_neg.parquet` are already fully
      correct-only / wrong-only.  In that case the filter step is a no-op
      (all rows already satisfy the requested mode) but we still run prompt
      normalization so the output is consumable by verl's SFTDataset.
    - If the requested mode conflicts with the file's actual content
      (e.g. --only_wrong 1 on sft_pos.parquet), we fail fast instead of
      silently emitting an empty parquet.

Output:
    a new parquet written to --output_parquet, kept as-is (prompt/response +
    provenance columns).  This is exactly what verl's SFTDataset expects
    (it only reads `prompt` and `response`).
"""

import argparse
import os

import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_parquet", required=True,
                    help="rollout parquet with is_correct column, or a paired "
                         "parquet from build_pairs_from_rollout.py "
                         "(sft_pos.parquet / sft_neg.parquet).")
    ap.add_argument("--output_parquet", required=True,
                    help="filtered parquet used as SFT input")
    ap.add_argument("--only_correct", type=int, default=1,
                    help="1 -> keep only is_correct==True (default), 0 -> do not force correct-only")
    ap.add_argument("--only_wrong", type=int, default=0,
                    help="1 -> keep only is_correct==False (SFT-on-wrong-only). "
                         "Must be used with --only_correct 0.")
    args = ap.parse_args()

    if args.only_correct and args.only_wrong:
        raise ValueError("--only_correct and --only_wrong cannot both be 1; "
                         "pass --only_correct 0 --only_wrong 1 for wrong-only.")

    print(f"[filter] loading {args.input_parquet}")
    df = pd.read_parquet(args.input_parquet)
    print(f"[filter] input rows = {len(df)}, cols = {list(df.columns)}")

    # ------------------------------------------------------------------
    # Detect the "already paired" case: input is one side of a pair produced
    # by build_pairs_from_rollout.py (all rows share the same is_correct
    # value, OR the parquet has a `pair_id` column).
    # ------------------------------------------------------------------
    is_paired_input = "pair_id" in df.columns
    file_is_all_correct = None  # True / False / None (mixed / unknown)
    if "is_correct" in df.columns and len(df) > 0:
        _bool_ic = df["is_correct"].astype(bool)
        if _bool_ic.all():
            file_is_all_correct = True
        elif (~_bool_ic).all():
            file_is_all_correct = False

    if is_paired_input:
        print(f"[filter] detected paired-SFT input (has `pair_id`)")
    if file_is_all_correct is True:
        print(f"[filter] input is uniformly is_correct=True (looks like sft_pos.parquet)")
    elif file_is_all_correct is False:
        print(f"[filter] input is uniformly is_correct=False (looks like sft_neg.parquet)")

    # ------------------------------------------------------------------
    # Fail fast on mode/file conflicts to avoid silently producing 0 rows.
    # ------------------------------------------------------------------
    if args.only_correct and file_is_all_correct is False:
        raise ValueError(
            "--only_correct 1 requested, but the input parquet is uniformly "
            "is_correct=False (looks like sft_neg.parquet). Refusing to "
            "produce an empty parquet. Point --input_parquet at the pos file "
            "or a mixed rollout parquet instead."
        )
    if args.only_wrong and file_is_all_correct is True:
        raise ValueError(
            "--only_wrong 1 requested, but the input parquet is uniformly "
            "is_correct=True (looks like sft_pos.parquet). Refusing to "
            "produce an empty parquet. Point --input_parquet at the neg file "
            "or a mixed rollout parquet instead."
        )

    if args.only_correct:
        if "is_correct" not in df.columns:
            raise ValueError("`is_correct` column not found; can't filter correct samples.")
        before = len(df)
        df = df[df["is_correct"].astype(bool)].reset_index(drop=True)
        print(f"[filter] is_correct=True: kept {len(df)}/{before}")
    elif args.only_wrong:
        if "is_correct" not in df.columns:
            raise ValueError("`is_correct` column not found; can't filter wrong samples.")
        before = len(df)
        df = df[~df["is_correct"].astype(bool)].reset_index(drop=True)
        print(f"[filter] is_correct=False: kept {len(df)}/{before}")
    else:
        print(f"[filter] no filter applied, keeping all {len(df)} rows")

    if len(df) == 0:
        raise RuntimeError("after filtering, 0 rows left. aborting.")

    for col in ("prompt", "response"):
        if col not in df.columns:
            raise ValueError(f"required column `{col}` missing from input parquet.")

    # ------------------------------------------------------------------
    # Normalize `prompt` into a plain string, because verl's SFTDataset does:
    #     prompt_chat = [{"role": "user", "content": prompt}]
    # i.e. it expects `prompt` to already be a raw user string. Our rollout
    # parquet (from simple_infer_dapo17k.py) instead stores prompt as a
    # list/np.ndarray of {"role","content"} chat messages -- if fed as-is,
    # apply_chat_template would either crash or stringify the whole array.
    # The paired parquets (sft_pos/neg.parquet) share the same prompt shape,
    # so this normalization also covers the new format.
    #
    # Strategy: pick the LAST user-turn's content; fall back to concatenating
    # all turn contents; final fallback is str(prompt).
    # ------------------------------------------------------------------
    def _prompt_to_str(p):
        # already a plain string -> keep
        if isinstance(p, str):
            return p
        # np.ndarray -> list
        if hasattr(p, "tolist") and not isinstance(p, (bytes, bytearray)):
            try:
                p = p.tolist()
            except Exception:
                pass
        if isinstance(p, list):
            # last user turn wins
            for turn in reversed(p):
                if isinstance(turn, dict) and turn.get("role") == "user":
                    return str(turn.get("content", ""))
            # otherwise concat all turn contents
            parts = []
            for turn in p:
                if isinstance(turn, dict):
                    parts.append(str(turn.get("content", "")))
                else:
                    parts.append(str(turn))
            return "\n".join(parts)
        return str(p)

    n_before_norm = len(df)
    df["prompt"] = df["prompt"].apply(_prompt_to_str)
    df["response"] = df["response"].astype(str)
    # sanity: drop rows with empty prompt/response after normalization
    non_empty = (df["prompt"].str.len() > 0) & (df["response"].str.len() > 0)
    n_dropped = int((~non_empty).sum())
    if n_dropped > 0:
        print(f"[filter] dropping {n_dropped} rows with empty prompt/response after normalization")
        df = df[non_empty].reset_index(drop=True)
    print(f"[filter] normalized prompts to plain str: {len(df)}/{n_before_norm} rows kept")
    if len(df) == 0:
        raise RuntimeError("after prompt normalization, 0 rows left. aborting.")

    # ------------------------------------------------------------------
    # Final safety asserts when --only_correct 1: after filtering, EVERY
    # surviving row must actually be a correct rollout with a well-formed
    # thinking-style response. This turns a silent data-quality regression
    # into a loud crash before SFT starts consuming the parquet.
    # ------------------------------------------------------------------
    if args.only_correct:
        if "is_correct" in df.columns:
            n_bad = int((~df["is_correct"].astype(bool)).sum())
            if n_bad > 0:
                raise RuntimeError(
                    f"[filter][assert] {n_bad} rows with is_correct=False leaked "
                    f"into the output parquet. Refusing to write."
                )
        # response should contain </think> and 'Answer:' (thinking + final answer)
        n_no_close  = int((~df["response"].str.contains("</think>")).sum())
        _ans_re     = r"Answer\s*:"
        n_no_answer = int((~df["response"].str.contains(_ans_re, regex=True)).sum())
        if n_no_close > 0 or n_no_answer > 0:
            raise RuntimeError(
                f"[filter][assert] {n_no_close} rows missing '</think>' and "
                f"{n_no_answer} rows missing 'Answer:' in response. "
                f"Refusing to write."
            )
        print(f"[filter][assert] all {len(df)} rows have is_correct=True, "
              f"'</think>' and 'Answer:' in response.")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_parquet)) or ".", exist_ok=True)
    df.to_parquet(args.output_parquet, index=False)
    print(f"[filter] wrote {len(df)} rows -> {args.output_parquet}")

if __name__ == "__main__":
    main()
