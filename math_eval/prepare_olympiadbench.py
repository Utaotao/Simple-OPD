"""Prepare Hothan/OlympiadBench (English math) for eval_math.py.

Pulls the *open-ended, text-only, English math, competition* subset
(`OE_TO_maths_en_COMP`) from the HuggingFace dataset
`Hothan/OlympiadBench` and converts it to the per-line schema that
`eval_math.py` expects:

    {"problem": <question>, "answer": <ground_truth_string>}

Only single-answer, non-theorem-proof problems are kept, because
`math_verify` (the verifier used by `eval_math.py`) compares a single
boxed expression against a single ground-truth boxed expression.
Multi-answer / theorem-proof items are skipped and reported.

Output:
    ${GOPD_ROOT}/data/olympiadbench/test.jsonl
"""
import argparse
import json
import os
import re

GOPD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DST_DIR = os.path.join(GOPD_ROOT, "data", "olympiadbench")
DST_FILE = os.path.join(DST_DIR, "test.jsonl")

DEFAULT_SUBSET = "OE_TO_maths_en_COMP"
DEFAULT_SPLIT = "train"  # OlympiadBench publishes everything under `train`

BOXED_RE = re.compile(r"^\s*\\boxed\{(.*)\}\s*$", re.DOTALL)


def strip_boxed(s: str) -> str:
    """Remove a single outer \\boxed{...} wrapper if present."""
    if not isinstance(s, str):
        return s
    m = BOXED_RE.match(s.strip())
    if m:
        return m.group(1).strip()
    return s.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subset",
        default=DEFAULT_SUBSET,
        help="HF subset/config name. Default: OE_TO_maths_en_COMP "
             "(open-ended, text-only, math, English, competition).",
    )
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--out", default=DST_FILE)
    parser.add_argument(
        "--keep-multi-answer",
        action="store_true",
        help="Keep is_multiple_answer items (default: skip them, "
             "since math_verify cannot compare multi-answer tuples).",
    )
    parser.add_argument(
        "--keep-theorem-proof",
        action="store_true",
        help="Keep Theorem proof items (default: skip them, since "
             "they have no closed-form ground truth).",
    )
    args = parser.parse_args()

    from datasets import load_dataset

    print(f"[load] Hothan/OlympiadBench  subset={args.subset}  split={args.split}")
    ds = load_dataset("Hothan/OlympiadBench", args.subset, split=args.split)
    print(f"[load] {len(ds)} raw samples")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    n_kept = 0
    n_skip_no_ans = 0
    n_skip_multi = 0
    n_skip_proof = 0
    n_skip_other = 0

    with open(args.out, "w", encoding="utf-8") as f:
        for item in ds:
            question = item.get("question") or ""
            final_answer = item.get("final_answer")
            qtype = item.get("question_type") or ""
            is_multi = bool(item.get("is_multiple_answer"))

            # Theorem proof has no closed-form answer.
            if "proof" in qtype.lower() or "theorem" in qtype.lower():
                if not args.keep_theorem_proof:
                    n_skip_proof += 1
                    continue

            if is_multi and not args.keep_multi_answer:
                n_skip_multi += 1
                continue

            if not question or not final_answer:
                n_skip_no_ans += 1
                continue

            # final_answer is a list of latex strings; for single-answer
            # items take the first. For kept multi-answer items, join
            # with commas inside a single boxed expression (math_verify
            # will at least try to parse this).
            if isinstance(final_answer, (list, tuple)):
                if len(final_answer) == 0:
                    n_skip_no_ans += 1
                    continue
                if is_multi and args.keep_multi_answer:
                    answer = ",".join(strip_boxed(str(x)) for x in final_answer)
                else:
                    answer = strip_boxed(str(final_answer[0]))
            else:
                answer = strip_boxed(str(final_answer))

            if not answer:
                n_skip_no_ans += 1
                continue

            rec = {
                "problem": question,
                "answer": answer,
                # Keep the original metadata around for debugging /
                # downstream re-scoring; eval_math.py ignores extra keys.
                "subject": item.get("subject"),
                "language": item.get("language"),
                "question_type": qtype,
                "answer_type": item.get("answer_type"),
                "is_multiple_answer": is_multi,
                "unit": item.get("unit"),
                "source": f"Hothan/OlympiadBench:{args.subset}",
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_kept += 1

    print(f"[done] wrote {n_kept} samples -> {args.out}")
    print(f"       skipped: theorem_proof={n_skip_proof}  "
          f"multi_answer={n_skip_multi}  "
          f"no_answer={n_skip_no_ans}  other={n_skip_other}")


if __name__ == "__main__":
    main()
