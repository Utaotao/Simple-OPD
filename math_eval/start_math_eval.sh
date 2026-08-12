#!/usr/bin/env bash
# Standalone math evaluation launcher.
#
# Evaluate one model:
#   MODEL_PATH=/path/to/model bash math_eval/start_math_eval.sh
#
# Evaluate all OPD checkpoints produced by start_opd.sh:
#   OPD_ROOT=/path/to/opd-run bash math_eval/start_math_eval.sh
#
# MODEL_PATH and OPD_ROOT are mutually exclusive.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GOPD_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EVAL_IMPL=${EVAL_IMPL:-"${SCRIPT_DIR}/run_eval_after_opd.sh"}
PYTHON_BIN=${PYTHON_BIN:-python}
EVAL_ROOT=${EVAL_ROOT:-"${GOPD_ROOT}/outputs/math_eval"}
DATA_ROOT=${DATA_ROOT:-"${GOPD_ROOT}/../G-OPD-Training-Data"}
MATH500_FILE=${MATH500_FILE:-"${DATA_ROOT}/MATH-500/test.jsonl"}
BENCHMARKS=${BENCHMARKS:-"aime24 aime25 amc23 math500"}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
RUN_TS=${RUN_TS:-$(date +%Y%m%d_%H%M%S)}

if [ -n "${VERL_CONDA_PREFIX:-}" ]; then
    export PATH="${VERL_CONDA_PREFIX}/bin:${PATH}"
    export LD_LIBRARY_PATH="${VERL_CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
unset BASH_ENV

if [ -n "${MODEL_PATH:-}" ] && [ -n "${OPD_ROOT:-}" ]; then
    echo "[math-eval][ERROR] set only one of MODEL_PATH or OPD_ROOT"
    exit 2
fi
if [ -z "${MODEL_PATH:-}" ] && [ -z "${OPD_ROOT:-}" ]; then
    echo "[math-eval][ERROR] set MODEL_PATH or OPD_ROOT"
    exit 2
fi
if [ ! -f "${EVAL_IMPL}" ]; then
    echo "[math-eval][ERROR] evaluator not found: ${EVAL_IMPL}"
    exit 1
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "[math-eval][ERROR] Python interpreter not found: ${PYTHON_BIN}"
    exit 1
fi
if ! [[ "${GPUS}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "[math-eval][ERROR] GPUS must be a comma-separated GPU list, got: ${GPUS}"
    exit 2
fi
mkdir -p "${EVAL_ROOT}"

case " ${BENCHMARKS//,/ } " in
    *" math500 "*)
        if [ ! -f "${MATH500_FILE}" ]; then
            echo "[math-eval][ERROR] math500 selected but MATH500_FILE is missing: ${MATH500_FILE}"
            exit 1
        fi
        ;;
esac

declare -a MODELS=()
declare -a TAGS=()
if [ -n "${MODEL_PATH:-}" ]; then
    MODELS+=("${MODEL_PATH}")
    DEFAULT_TAG="$(basename "${MODEL_PATH%/}")"
    if [ "${DEFAULT_TAG}" = "merged" ]; then
        DEFAULT_TAG="$(basename "$(dirname "${MODEL_PATH%/}")")"
    fi
    TAGS+=("${MODEL_TAG:-${DEFAULT_TAG}}")
else
    if [ ! -d "${OPD_ROOT}" ]; then
        echo "[math-eval][ERROR] OPD_ROOT not found: ${OPD_ROOT}"
        exit 1
    fi
    while IFS= read -r STEP_DIR; do
        DIR="${STEP_DIR}/merged"
        [ -d "${DIR}" ] || continue
        [ -f "${DIR}/config.json" ] || continue
        MODELS+=("${DIR}")
        TAGS+=("$(basename "${OPD_ROOT%/}")_$(basename "${STEP_DIR}")")
    done < <(find "${OPD_ROOT}" -mindepth 1 -maxdepth 1 -type d -name 'step[0-9]*' -print \
        | awk -F/ '$NF ~ /^step[0-9]+$/' | sort -V)
    if [ "${#MODELS[@]}" -eq 0 ]; then
        echo "[math-eval][ERROR] no step*/merged/config.json found under ${OPD_ROOT}"
        exit 1
    fi
fi

# A local path must contain an HF config. Hub IDs are checked by Transformers.
for MODEL in "${MODELS[@]}"; do
    if [[ "${MODEL}" == /* || "${MODEL}" == ./* || "${MODEL}" == ../* ]]; then
        if [ ! -f "${MODEL%/}/config.json" ]; then
            echo "[math-eval][ERROR] local model is not an HF checkpoint: ${MODEL}"
            exit 1
        fi
    fi
done

echo "[math-eval] models      = ${#MODELS[@]}"
echo "[math-eval] benchmarks = ${BENCHMARKS}"
echo "[math-eval] gpus       = ${GPUS}"
echo "[math-eval] output     = ${EVAL_ROOT}"

FAIL=0
for INDEX in "${!MODELS[@]}"; do
    MODEL="${MODELS[$INDEX]}"
    TAG="${TAGS[$INDEX]}"
    SAFE_TAG="$(printf '%s' "${TAG}" | tr -c 'A-Za-z0-9._-' '_')"
    EXPECTED_RUN_DIR="${EVAL_ROOT}/${SAFE_TAG}/all_${RUN_TS}"
    echo "[math-eval] evaluating ${SAFE_TAG}: ${MODEL}"
    set +e
    MODEL_PATH="${MODEL}" \
    MODEL_TAG="${SAFE_TAG}" \
    RUN_TS="${RUN_TS}" \
    EVAL_ROOT="${EVAL_ROOT}" \
    BENCHMARKS="${BENCHMARKS}" \
    GPUS="${GPUS}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    DATA_ROOT="${DATA_ROOT}" \
    MATH500_FILE="${MATH500_FILE}" \
        bash "${EVAL_IMPL}"
    RC=$?
    set -e
    if [ "${RC}" -ne 0 ]; then
        FAIL=$((FAIL + 1))
        echo "[math-eval][WARN] ${SAFE_TAG} failed with rc=${RC}"
    elif [ ! -s "${EXPECTED_RUN_DIR}/summary.csv" ]; then
        FAIL=$((FAIL + 1))
        echo "[math-eval][WARN] ${SAFE_TAG} produced no summary.csv"
    fi
done

echo "[math-eval] complete: total=${#MODELS[@]}, failed=${FAIL}"
echo "MATH_EVAL_ROOT=${EVAL_ROOT}"
[ "${FAIL}" -eq 0 ]
