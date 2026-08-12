#!/bin/bash

set -e
set -x
export PYTHONUNBUFFERED=1
if [ -n "${VERL_CONDA_PREFIX:-}" ]; then
    export LD_LIBRARY_PATH="${VERL_CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

# ===== Project roots =====
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GOPD_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python}
EVAL_SCRIPT=${GOPD_ROOT}/math_eval/eval_math.py
PREPARE_OLYMPIAD=${GOPD_ROOT}/math_eval/prepare_olympiadbench.py

# Output root (NOT under G-OPD repo)
EVAL_ROOT=${EVAL_ROOT:-"${GOPD_ROOT}/outputs/eval"}

# ===== Model under inference =====
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-1.7B}
MODEL_TAG=${MODEL_TAG:-$(basename "${MODEL_PATH}")}
RUN_TS=${RUN_TS:-$(date +%Y%m%d_%H%M%S)}

# Put the merged "all benchmarks" run under its own subdir so it doesn't
# collide with prior runs from the two single-suite scripts.
RUN_DIR=${EVAL_ROOT}/${MODEL_TAG}/all_${RUN_TS}
mkdir -p "${RUN_DIR}"

LOG_FILE=${RUN_DIR}/run.log
SUMMARY_FILE=${RUN_DIR}/summary.txt
exec > >(tee -a "${LOG_FILE}") 2>&1

# ===== Datasets to evaluate (single TP=8 pass over all of them) =====
# IMPORTANT: keep DATASETS, INPUTS, N_LIST in lockstep (same length, same order).
declare -a DATASETS=(
    "aime24"
    "aime25"
    "amc23"
    "math500"
)
declare -a INPUTS=(
    "${GOPD_ROOT}/data/aime24/test.jsonl"
    "${GOPD_ROOT}/data/aime25/test.jsonl"
    "${GOPD_ROOT}/data/amc23/test.jsonl"
    "${GOPD_ROOT}/MATH-500/test.jsonl"
)
# Per-dataset k in pass@k. Competition sets (~30-40 problems) need n=16
# for a meaningful pass@k; MATH-500 has hundreds of problems and n=4 is plenty.
declare -a N_LIST=(
    16   # aime24
    16   # aime25
    16   # amc23
    4    # math500
)

# =====================================================================
# Optional: pick a SUBSET of benchmarks to evaluate, via either
#   (a) positional CLI args :  bash run_eval_all_base.sh hmmt25 aime24 math500
#   (b) env var BENCHMARKS   :  BENCHMARKS="hmmt25,aime24" bash run_eval_all_base.sh
# Both forms accept comma- or whitespace-separated names. If neither is
# provided, ALL datasets above are evaluated (default behaviour).
# Execution order always follows the canonical order in DATASETS above,
# regardless of the order in which the user passes the names.
# =====================================================================
declare -a USER_REQUESTED=()
if [[ -n "${BENCHMARKS:-}" ]]; then
    IFS=', ' read -r -a _BM_FROM_ENV <<< "${BENCHMARKS}"
    USER_REQUESTED+=("${_BM_FROM_ENV[@]}")
fi
if [[ $# -gt 0 ]]; then
    USER_REQUESTED+=("$@")
fi

if [[ ${#USER_REQUESTED[@]} -gt 0 ]]; then
    declare -a UNKNOWN=()
    for req in "${USER_REQUESTED[@]}"; do
        [[ -z "${req}" ]] && continue
        found=0
        for ds in "${DATASETS[@]}"; do
            if [[ "${ds}" == "${req}" ]]; then found=1; break; fi
        done
        [[ ${found} -eq 0 ]] && UNKNOWN+=("${req}")
    done
    if [[ ${#UNKNOWN[@]} -gt 0 ]]; then
        echo "[fatal] unknown benchmark name(s): ${UNKNOWN[*]}"
        echo "[fatal] supported: ${DATASETS[*]}"
        exit 3
    fi

    declare -a _F_DATASETS=()
    declare -a _F_INPUTS=()
    declare -a _F_NLIST=()
    for idx in "${!DATASETS[@]}"; do
        ds="${DATASETS[$idx]}"
        for req in "${USER_REQUESTED[@]}"; do
            if [[ "${ds}" == "${req}" ]]; then
                _F_DATASETS+=("${ds}")
                _F_INPUTS+=("${INPUTS[$idx]}")
                _F_NLIST+=("${N_LIST[$idx]}")
                break
            fi
        done
    done
    DATASETS=("${_F_DATASETS[@]}")
    INPUTS=("${_F_INPUTS[@]}")
    N_LIST=("${_F_NLIST[@]}")
    echo "[info] running on requested subset: ${DATASETS[*]}"
else
    echo "[info] no subset specified; running all benchmarks: ${DATASETS[*]}"
fi

# Auto-prepare OlympiadBench jsonl if missing (one-time HF download).
# Only attempt this if olympiadbench is actually in the selected subset.
NEED_OLYMPIAD=0
for ds in "${DATASETS[@]}"; do
    if [[ "${ds}" == "olympiadbench" ]]; then NEED_OLYMPIAD=1; break; fi
done
if [[ ${NEED_OLYMPIAD} -eq 1 ]]; then
    OLYMPIAD_JSONL="${GOPD_ROOT}/data/olympiadbench/test.jsonl"
    if [[ ! -f "${OLYMPIAD_JSONL}" ]]; then
        echo "[prepare] OlympiadBench jsonl not found, generating once..."
        "${PYTHON_BIN}" "${PREPARE_OLYMPIAD}" \
            --out "${OLYMPIAD_JSONL}" \
            || echo "[warn] prepare_olympiadbench.py failed; olympiadbench will be skipped."
    fi
fi

# ===== Sampling params (shared by all datasets; matches both originals) =====
MAX_TOKENS=${MAX_TOKENS:-16384}
TEMPERATURE=${TEMPERATURE:-0.7}
TOP_P=${TOP_P:-0.8}
TOP_K=${TOP_K:-20}
SEED=${SEED:-42}
# Take the max of the two original scripts' MAX_NUM_SEQS. This is the
# vLLM engine's concurrency cap (set at LLM construction time, so it
# MUST be a single value). 4096 is safely within the TP=8 KV-cache
# budget at 16k tokens; small datasets (e.g. aime24 with 30*16=480 seqs)
# simply don't fill it, which is fine.
MAX_NUM_SEQS=${MAX_NUM_SEQS:-4096}

# All 8 GPUs on THIS pod go to one TP=8 process. Eval runs on rank-0 pod only,
# not distributed across two pods -- vLLM inference doesn't need multi-node.
GPUS=${GPUS:-0,1,2,3,4,5,6,7}

echo "=========================================================="
echo "[run_dir]      ${RUN_DIR}"
echo "[model]        ${MODEL_PATH}"
echo "[gpus]         ${GPUS}  (TP=$(echo "${GPUS}" | tr ',' '\n' | wc -l))"
echo "[max_tokens]   ${MAX_TOKENS}"
echo "[max_num_seqs] ${MAX_NUM_SEQS}"
echo "[sampling]     T=${TEMPERATURE} top_p=${TOP_P} top_k=${TOP_K} seed=${SEED}"
echo "[per-dataset n]"
for idx in "${!DATASETS[@]}"; do
    printf "                 %-15s n=%d\n" "${DATASETS[$idx]}" "${N_LIST[$idx]}"
done
echo "=========================================================="

# =====================================================================
# Build the multi-dataset --input_file / --output_file / --n argument
# lists, skipping any dataset whose input file is missing.
# =====================================================================
declare -a EFF_DATASETS=()
declare -a EFF_INPUTS=()
declare -a EFF_OUTPUTS=()
declare -a EFF_NS=()
for idx in "${!DATASETS[@]}"; do
    DS="${DATASETS[$idx]}"
    INPUT="${INPUTS[$idx]}"
    N_DS="${N_LIST[$idx]}"
    if [[ ! -f "${INPUT}" ]]; then
        echo "[skip] ${DS}: input file not found -> ${INPUT}"
        continue
    fi
    EFF_DATASETS+=("${DS}")
    EFF_INPUTS+=("${INPUT}")
    EFF_OUTPUTS+=("${RUN_DIR}/${DS}.jsonl")
    EFF_NS+=("${N_DS}")
done

if [[ ${#EFF_DATASETS[@]} -eq 0 ]]; then
    echo "[fatal] no datasets to evaluate; check input paths."
    exit 1
fi

ALL_LOG="${RUN_DIR}/all_datasets.log"

echo ""
echo "########## Single TP=8 run over ${#EFF_DATASETS[@]} datasets ##########"
echo "  datasets : ${EFF_DATASETS[*]}"
echo "  n values : ${EFF_NS[*]}"
echo "  log      : ${ALL_LOG}"
echo "----------------------------------------------------------"

set +e
CUDA_VISIBLE_DEVICES=${GPUS} \
    "${PYTHON_BIN}" "${EVAL_SCRIPT}" \
        --input_file "${EFF_INPUTS[@]}" \
        --output_file "${EFF_OUTPUTS[@]}" \
        --model_path "${MODEL_PATH}" \
        --max_tokens ${MAX_TOKENS} \
        --temperature ${TEMPERATURE} \
        --top_p ${TOP_P} \
        --top_k ${TOP_K} \
        --max_num_seqs ${MAX_NUM_SEQS} \
        --n "${EFF_NS[@]}" \
        --begin_idx -1 \
        --end_idx -1 \
        2>&1 | tee "${ALL_LOG}"
RC=${PIPESTATUS[0]}
set -e
echo "[done] eval_math.py exit=${RC}"
FAIL=0
if [[ ${RC} -ne 0 ]]; then
    FAIL=$((FAIL+1))
fi

# Hard sanity check: did the python side actually produce any output?
if [[ ! -s "${ALL_LOG}" ]] || ! grep -qE "^===== dataset\[" "${ALL_LOG}"; then
    echo "[fatal] eval_math.py never produced a dataset block."
    echo "[fatal] dump of ${ALL_LOG}:"
    sed -n '1,20p' "${ALL_LOG}" || true
    echo "[fatal] aborting before generating a misleading summary.csv."
    exit 2
fi

# =====================================================================
# Split the merged log back into per-dataset logs so the summary /
# CSV step below can reuse the existing grep-based parser.
# =====================================================================
"${PYTHON_BIN}" - <<PYEOF
import os
import re

all_log  = "${ALL_LOG}"
run_dir  = "${RUN_DIR}"
ds_names = "${EFF_DATASETS[*]}".split()

with open(all_log, "r", encoding="utf-8", errors="replace") as f:
    text = f.read()

pat = re.compile(r"^===== dataset\[(\d+)\]: .* =====$", re.M)
matches = list(pat.finditer(text))
if not matches:
    print(f"[warn] no dataset markers found in {all_log}; "
          f"per-dataset logs will not be created.")
else:
    for i, m in enumerate(matches):
        idx = int(m.group(1))
        if idx >= len(ds_names):
            continue
        start = m.end()
        end   = matches[i+1].start() if i+1 < len(matches) else len(text)
        block = text[start:end]
        ds_log_path = os.path.join(run_dir, ds_names[idx] + ".log")
        with open(ds_log_path, "w", encoding="utf-8") as f:
            f.write(block)
        print(f"[split] wrote {ds_log_path}")
PYEOF

echo "[done] all inference jobs finished (failures=${FAIL})"

# =====================================================================
# Per-dataset n lookup helper for the summary section. Bash 3 doesn't
# guarantee associative arrays, so use a parallel-array lookup func.
# =====================================================================
n_for_dataset() {
    local target="$1"
    for idx in "${!DATASETS[@]}"; do
        if [[ "${DATASETS[$idx]}" == "${target}" ]]; then
            echo "${N_LIST[$idx]}"
            return 0
        fi
    done
    echo ""
}

# =====================================================================
# Aggregate pass@k / accuracy / avg_length from each dataset's log
# =====================================================================
echo ""
echo "==================== SUMMARY ====================" | tee "${SUMMARY_FILE}"
printf "%-15s | %-3s | %-10s | %-10s | %-12s\n" "dataset" "n" "acc(avg@k)" "pass@k" "avg_length" | tee -a "${SUMMARY_FILE}"
echo "----------------------------------------------------------" | tee -a "${SUMMARY_FILE}"

for DS in "${DATASETS[@]}"; do
    DS_LOG="${RUN_DIR}/${DS}.log"
    DS_N=$(n_for_dataset "${DS}")
    if [[ ! -f "${DS_LOG}" ]]; then
        printf "%-15s | %-3s | (no log)\n" "${DS}" "${DS_N:-?}" | tee -a "${SUMMARY_FILE}"
        continue
    fi
    ACC=$(grep -E "^Accuracy:"   "${DS_LOG}" | tail -n1 | awk '{print $2}')
    PK=$(grep  -E "^passs?@k:"   "${DS_LOG}" | tail -n1 | awk '{print $2}')
    AL=$(grep  -E "^avg_length:" "${DS_LOG}" | tail -n1 | awk '{print $2}')
    printf "%-15s | %-3s | %-10s | %-10s | %-12s\n" "${DS}" "${DS_N:-?}" "${ACC:-NA}" "${PK:-NA}" "${AL:-NA}" | tee -a "${SUMMARY_FILE}"
done

# =====================================================================
# Wide-format CSV (one row per model run) for easy spreadsheet pasting.
# Column header includes the per-dataset n so it's unambiguous, e.g.
#   aime24_avg@16, aime24_pass@16, ..., math500_avg@4, math500_pass@4.
# Numbers are in percent (e.g. 42.50).
# A "competition" mean (aime/amc, n=16) and an overall mean are
# also emitted at the end of the row.
# =====================================================================
SUMMARY_CSV="${RUN_DIR}/summary.csv"
# Canonical CSV column order (matches spreadsheet template).
declare -a CSV_ORDER_ALL=("hmmt25" "hmmt24" "amc23" "aime25" "aime24" "math500" "olympiadbench")
declare -a COMP_ALL=("hmmt25" "hmmt24" "amc23" "aime25" "aime24")
# Restrict CSV columns to whatever was actually evaluated this run,
# while preserving the canonical column order above.
declare -a CSV_DATASETS=()
declare -a COMP_DATASETS=()
for col in "${CSV_ORDER_ALL[@]}"; do
    for ds in "${DATASETS[@]}"; do
        if [[ "${ds}" == "${col}" ]]; then CSV_DATASETS+=("${col}"); break; fi
    done
done
for col in "${COMP_ALL[@]}"; do
    for ds in "${DATASETS[@]}"; do
        if [[ "${ds}" == "${col}" ]]; then COMP_DATASETS+=("${col}"); break; fi
    done
done

HEADER="model"
for DS in "${CSV_DATASETS[@]}"; do
    DS_N=$(n_for_dataset "${DS}")
    HEADER="${HEADER},${DS}_avg@${DS_N},${DS}_pass@${DS_N}"
done
HEADER="${HEADER},comp-mean-avg,comp-mean-pass,all-mean-avg,all-mean-pass"
echo "${HEADER}" > "${SUMMARY_CSV}"

ROW="${MODEL_TAG}"
SUM_AVG_COMP=0; SUM_PK_COMP=0; N_OK_AVG_COMP=0; N_OK_PK_COMP=0
SUM_AVG_ALL=0;  SUM_PK_ALL=0;  N_OK_AVG_ALL=0;  N_OK_PK_ALL=0

is_competition() {
    local ds="$1"
    for x in "${COMP_DATASETS[@]}"; do
        [[ "${x}" == "${ds}" ]] && return 0
    done
    return 1
}

for DS in "${CSV_DATASETS[@]}"; do
    DS_LOG="${RUN_DIR}/${DS}.log"
    AVG_PCT=""
    PK_PCT=""
    if [[ -f "${DS_LOG}" ]]; then
        ACC=$(grep -E "^Accuracy:" "${DS_LOG}" | tail -n1 | awk '{print $2}')
        PK=$(grep  -E "^passs?@k:" "${DS_LOG}" | tail -n1 | awk '{print $2}')
        if [[ -n "${ACC}" ]]; then
            AVG_PCT=$(awk -v x="${ACC}" 'BEGIN{ printf "%.2f", x*100 }')
            SUM_AVG_ALL=$(awk -v s="${SUM_AVG_ALL}" -v x="${AVG_PCT}" 'BEGIN{ printf "%.6f", s+x }')
            N_OK_AVG_ALL=$((N_OK_AVG_ALL+1))
            if is_competition "${DS}"; then
                SUM_AVG_COMP=$(awk -v s="${SUM_AVG_COMP}" -v x="${AVG_PCT}" 'BEGIN{ printf "%.6f", s+x }')
                N_OK_AVG_COMP=$((N_OK_AVG_COMP+1))
            fi
        fi
        if [[ -n "${PK}" ]]; then
            PK_PCT=$(awk -v x="${PK}" 'BEGIN{ printf "%.2f", x*100 }')
            SUM_PK_ALL=$(awk -v s="${SUM_PK_ALL}" -v x="${PK_PCT}" 'BEGIN{ printf "%.6f", s+x }')
            N_OK_PK_ALL=$((N_OK_PK_ALL+1))
            if is_competition "${DS}"; then
                SUM_PK_COMP=$(awk -v s="${SUM_PK_COMP}" -v x="${PK_PCT}" 'BEGIN{ printf "%.6f", s+x }')
                N_OK_PK_COMP=$((N_OK_PK_COMP+1))
            fi
        fi
    fi
    ROW="${ROW},${AVG_PCT:-NA},${PK_PCT:-NA}"
done

mean_or_na() {
    local s="$1"; local n="$2"
    if [[ "${n}" -gt 0 ]]; then
        awk -v s="${s}" -v n="${n}" 'BEGIN{ printf "%.3f", s/n }'
    else
        echo "NA"
    fi
}
MEAN_AVG_COMP=$(mean_or_na "${SUM_AVG_COMP}" "${N_OK_AVG_COMP}")
MEAN_PK_COMP=$(mean_or_na  "${SUM_PK_COMP}"  "${N_OK_PK_COMP}")
MEAN_AVG_ALL=$(mean_or_na  "${SUM_AVG_ALL}"  "${N_OK_AVG_ALL}")
MEAN_PK_ALL=$(mean_or_na   "${SUM_PK_ALL}"   "${N_OK_PK_ALL}")

ROW="${ROW},${MEAN_AVG_COMP},${MEAN_PK_COMP},${MEAN_AVG_ALL},${MEAN_PK_ALL}"
echo "${ROW}" >> "${SUMMARY_CSV}"

echo ""
echo "==================== CSV ===================="
cat "${SUMMARY_CSV}"
echo "=============================================="

echo ""
echo "==================== FINISHED ===================="
echo "All artefacts saved under: ${RUN_DIR}"
echo "  - per-dataset jsonl : ${RUN_DIR}/<DATASET>.jsonl"
echo "  - per-dataset log   : ${RUN_DIR}/<DATASET>.log"
echo "  - merged log        : ${ALL_LOG}"
echo "  - summary           : ${SUMMARY_FILE}"
echo "  - summary csv       : ${SUMMARY_CSV}"
echo "  - main log          : ${LOG_FILE}"
