#!/bin/bash

set -euo pipefail

# ---- lock the verl conda env for training / merging ----
export VERL_CONDA_PREFIX=/root/miniconda3/envs/verl
export PATH="${VERL_CONDA_PREFIX}/bin:${PATH}"
unset BASH_ENV

# ---- required inputs (all overridable) ----
export project_root=${project_root:-your/absolute/path/to/project_root}
export MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-1.7B-Base}
export TEACHER=${TEACHER:-RedKiKi/Qwen3-8b-base-rl-dapo-17k}
export SFT_PARQUET=${SFT_PARQUET:-${project_root}/Simple-OPD/data/sft_pos.parquet}
export OUTPUT_ROOT=${OUTPUT_ROOT:-${project_root}/pipeline_runs}
export SFT_SAMPLE_MODE=${SFT_SAMPLE_MODE:-correct}
export CUTOFF_LEN=${CUTOFF_LEN:-16384}

# ---- SFT knobs ----
# Exactly ONE SFT ckpt at step 175, no SFT-side eval.
export SFT_MAX_STEPS=${SFT_MAX_STEPS:-175}
export SFT_SAVE_FREQ=${SFT_SAVE_FREQ:-175}
# LoRA settings for this variant. Alpha defaults to rank (alpha/r = 1) inside
# the internal SFT script, which also propagates alpha to
# merge_lora_into_base.py so the fold uses the exact training-time scaling.
export LORA_RANK=${LORA_RANK:-32}
export LORA_ALPHA=${LORA_ALPHA:-${LORA_RANK}}
export LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-all-linear}
# LoRA-appropriate LR (~20x the full-param 5e-6). Override with SFT_LR=... if
# you want to sweep this.
export SFT_LR=${SFT_LR:-5e-5}

# ---- OPD hyper-params ----
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-300}
export SAVE_FREQ=${SAVE_FREQ:-25}
export EVAL_STEP=${EVAL_STEP:-25}
export TEST_FREQ=${TEST_FREQ:-25}
STEPS_LIST=$(seq -s ' ' ${SAVE_FREQ} ${SAVE_FREQ} ${TOTAL_TRAINING_STEPS})

# ---- wandb ----
# Online mode + robustness knobs, identical to the full-param sft175 launcher.
export WANDB_API_KEY="your_wandb_api_key_here"
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_START_METHOD=${WANDB_START_METHOD:-thread}
export WANDB_CONSOLE=${WANDB_CONSOLE:-off}
export WANDB_INIT_TIMEOUT=${WANDB_INIT_TIMEOUT:-600}
export WANDB__SERVICE_WAIT=${WANDB__SERVICE_WAIT:-600}
export WANDB_HTTP_TIMEOUT=${WANDB_HTTP_TIMEOUT:-120}
export WANDB_DIR=${WANDB_DIR:-${project_root}/wandb_logs}
mkdir -p "${WANDB_DIR}"

# ---- sanity ----
if [ ! -f "${SFT_PARQUET}" ]; then
    echo "[sft175_opd300_lora32][ERROR] SFT_PARQUET not found: ${SFT_PARQUET}"
    exit 1
fi

# ---- resolve RUN_DIR ----
export TS=$(date +%Y%m%d_%H%M%S)
export MODEL_TAG_SAFE="$(printf '%s' "$(basename "${MODEL_PATH%/}")" | tr -c 'A-Za-z0-9._-' '_')"
export RUN_DIR="${OUTPUT_ROOT}/${MODEL_TAG_SAFE}-sft${SFT_MAX_STEPS}-lora${LORA_RANK}-opd${TOTAL_TRAINING_STEPS}_${TS}"
mkdir -p "${RUN_DIR}"
LOG="${RUN_DIR}/full_run.log"
exec > >(tee -a "${LOG}") 2>&1

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SFT_SCRIPT="${SCRIPT_DIR}/run_warmup_sft.sh"
OPD_SCRIPT="${SCRIPT_DIR}/run_qwen3-4b-opd-1.7b-dapo17k-single_teacher.sh"
MATH_EVAL_SCRIPT=/apdcephfs_cq8_1324356/share_1324356/utao/Simple-OPD/math_eval/run_eval_after_opd.sh
MODEL_MERGER_BIN="${VERL_CONDA_PREFIX}/bin/python"


# =============================================================================
# Stage 0: LoRA SFT (rank=32, 175 steps, no eval, only 1 ckpt at step 175)
# =============================================================================
SFT_STDOUT_LOG="${RUN_DIR}/sft_stdout.log"
set +e
SFT_PARQUET="${SFT_PARQUET}" \
MODEL_PATH="${MODEL_PATH}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
CUTOFF_LEN="${CUTOFF_LEN}" \
MAX_STEPS="${SFT_MAX_STEPS}" \
SAVE_FREQ="${SFT_SAVE_FREQ}" \
NPROC=8 \
SFT_SAMPLE_MODE="${SFT_SAMPLE_MODE}" \
SKIP_EVAL=1 \
KEEP_STEPS="${SFT_MAX_STEPS}" \
LORA_RANK="${LORA_RANK}" \
LORA_ALPHA="${LORA_ALPHA}" \
LORA_TARGET_MODULES="${LORA_TARGET_MODULES}" \
LR="${SFT_LR}" \
RUN_DIR="${RUN_DIR}" \
MODEL_TAG_SAFE="${MODEL_TAG_SAFE}" \
TS="${TS}" \
    bash "${SFT_SCRIPT}" 2>&1 | tee "${SFT_STDOUT_LOG}"
SFT_RC=${PIPESTATUS[0]}
set -e
if [ "${SFT_RC}" -ne 0 ]; then
    echo "[sft175_opd300_lora32][ERROR] SFT stage exited rc=${SFT_RC}, aborting."
    exit "${SFT_RC}"
fi

STUDENT_HF_DIR="${RUN_DIR}/hf/global_step_${SFT_MAX_STEPS}"
if [ ! -f "${STUDENT_HF_DIR}/config.json" ]; then
    echo "[sft175_opd300_lora32][ERROR] SFT merged HF not found: ${STUDENT_HF_DIR}"
    exit 1
fi
# The LoRA fold marker is written by merge_lora_into_base.py inside the
# internal SFT script. If it's missing we abort loudly: continuing would silently
# make OPD train on the un-adapted base model.
if [ ! -f "${STUDENT_HF_DIR}/.lora_merged" ]; then
    echo "[sft175_opd300_lora32][ERROR] LoRA fold marker .lora_merged is missing under ${STUDENT_HF_DIR}."
    echo "[sft175_opd300_lora32][ERROR] The HF weights are almost certainly the raw base model."
    echo "[sft175_opd300_lora32][ERROR] Refusing to launch OPD on an un-adapted student."
    exit 1
fi
echo "[sft175_opd300_lora32] SFT stage OK. LoRA folded. Student for OPD = ${STUDENT_HF_DIR}"

# =============================================================================
# Stage 1: OPD training (single job, saves 12 shard ckpts)
# =============================================================================
STUDENT_TAG="${MODEL_TAG_SAFE}-sft${SFT_MAX_STEPS}-lora${LORA_RANK}"
OPD_TRAIN_DIR="${RUN_DIR}/opd/train"
OPD_MERGED_ROOT="${RUN_DIR}/opd"      # step<N>/merged goes here
mkdir -p "${OPD_TRAIN_DIR}"

echo ""
echo "[sft175_opd300_lora32] >>> stage 1: OPD training (${TOTAL_TRAINING_STEPS} steps, save every ${SAVE_FREQ})"

set +e
STUDENT="${STUDENT_HF_DIR}" \
STUDENT_TAG="${STUDENT_TAG}" \
TEACHER="${TEACHER}" \
CHECKPOINTS_DIR="${OPD_TRAIN_DIR}" \
WANDB_DIR="${OPD_TRAIN_DIR}/wandb" \
RUN_NAME_SUFFIX="_${TS}" \
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS}" \
SAVE_FREQ="${SAVE_FREQ}" \
EVAL_STEP="${EVAL_STEP}" \
TEST_FREQ="${TEST_FREQ}" \
SKIP_MERGE=1 \
SKIP_EVAL=1 \
    bash "${OPD_SCRIPT}" 2>&1 | tee "${OPD_TRAIN_DIR}/run.log"
OPD_RC=${PIPESTATUS[0]}
set -e
if [ "${OPD_RC}" -ne 0 ]; then
    echo "[sft175_opd300_lora32][WARN] OPD training exited rc=${OPD_RC}. NOT aborting -- will still"
    echo "[sft175_opd300_lora32][WARN] attempt to merge/eval whatever shard ckpts landed on disk."
else
    echo "[sft175_opd300_lora32] OPD training OK."
fi

# =============================================================================
# Stage 2: merge every saved shard -> HF, land under opd/step<N>/merged/
# =============================================================================
echo ""
echo "[sft175_opd300_lora32] >>> stage 2: merge 12 OPD shards -> HF"

OPD_ANY_ACTOR=$(ls -1d "${OPD_TRAIN_DIR}"/*/global_step_*/actor 2>/dev/null | head -n1 || true)
if [ -z "${OPD_ANY_ACTOR}" ]; then
    echo "[sft175_opd300_lora32][ERROR] no global_step_*/actor found under ${OPD_TRAIN_DIR}/*/"
    echo "[sft175_opd300_lora32][ERROR] OPD training produced no shard checkpoints; nothing to eval."
    ls -la "${OPD_TRAIN_DIR}" || true
    exit 1
fi
OPD_RUN_NAME_DIR="$(dirname "$(dirname "${OPD_ANY_ACTOR}")")"
echo "[sft175_opd300_lora32] OPD run_name dir = ${OPD_RUN_NAME_DIR}"
echo "[sft175_opd300_lora32] shards present  = $(ls -1d "${OPD_RUN_NAME_DIR}"/global_step_*/actor 2>/dev/null | wc -l)"

MERGE_LOG="${RUN_DIR}/merge.log"
: > "${MERGE_LOG}"
MERGE_FAIL=()
for STEP in ${STEPS_LIST}; do
    SHARD_DIR="${OPD_RUN_NAME_DIR}/global_step_${STEP}/actor"
    MERGED_DIR="${OPD_MERGED_ROOT}/step${STEP}/merged"
    if [ ! -d "${SHARD_DIR}" ]; then
        echo "[sft175_opd300_lora32][WARN] shard missing for step=${STEP}: ${SHARD_DIR}" | tee -a "${MERGE_LOG}"
        MERGE_FAIL+=("${STEP}(no-shard)")
        continue
    fi
    if [ -f "${MERGED_DIR}/config.json" ] && \
       ( ls "${MERGED_DIR}"/*.safetensors >/dev/null 2>&1 || ls "${MERGED_DIR}"/pytorch_model*.bin >/dev/null 2>&1 ); then
        echo "[sft175_opd300_lora32] skip merge (already exists): ${MERGED_DIR}" | tee -a "${MERGE_LOG}"
        continue
    fi
    mkdir -p "${MERGED_DIR}"
    echo "[sft175_opd300_lora32] merging step=${STEP}: ${SHARD_DIR} -> ${MERGED_DIR}" | tee -a "${MERGE_LOG}"
    set +e
    ${MODEL_MERGER_BIN} -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "${SHARD_DIR}" \
        --target_dir "${MERGED_DIR}" \
        2>&1 | tee -a "${MERGE_LOG}"
    MRC=${PIPESTATUS[0]}
    set -e
    if [ "${MRC}" -ne 0 ]; then
        echo "[sft175_opd300_lora32][WARN] merge for step=${STEP} exited rc=${MRC}" | tee -a "${MERGE_LOG}"
        MERGE_FAIL+=("${STEP}(rc=${MRC})")
    fi
done

# =============================================================================
# Stage 3: math eval per merged ckpt (all 12)
# =============================================================================
echo ""
echo "[sft175_opd300_lora32] >>> stage 3: math eval on 12 OPD ckpts"

MATH_EVAL_ROOT="${RUN_DIR}/opd/math_eval"
mkdir -p "${MATH_EVAL_ROOT}"
MATH_FAIL=()

# ---- also math-eval the SFT ckpt (once, before the OPD sweep) ----
# Same interface as OPD eval below: run_eval_after_opd.sh just consumes
# MODEL_PATH/MODEL_TAG/RUN_TS/EVAL_ROOT, no dependency on SFT-vs-OPD.
SFT_ABBR="${MODEL_TAG_SAFE}-sft${SFT_MAX_STEPS}-lora${LORA_RANK}_sft_step${SFT_MAX_STEPS}"
SFT_EVAL_LOG="${RUN_DIR}/opd/math_eval_sft_step${SFT_MAX_STEPS}.log"
echo "[sft175_opd300_lora32] math eval SFT step=${SFT_MAX_STEPS}: ${STUDENT_HF_DIR}"
set +e
MODEL_PATH="${STUDENT_HF_DIR}" \
MODEL_TAG="${SFT_ABBR}" \
RUN_TS="${TS}_sft_step${SFT_MAX_STEPS}" \
EVAL_ROOT="${MATH_EVAL_ROOT}" \
    bash "${MATH_EVAL_SCRIPT}" 2>&1 | tee "${SFT_EVAL_LOG}"
SERC=${PIPESTATUS[0]}
set -e
if [ "${SERC}" -ne 0 ]; then
    echo "[sft175_opd300_lora32][WARN] math eval SFT step=${SFT_MAX_STEPS} exited rc=${SERC}"
    MATH_FAIL+=("sft${SFT_MAX_STEPS}(rc=${SERC})")
fi

for STEP in ${STEPS_LIST}; do
    MERGED_DIR="${OPD_MERGED_ROOT}/step${STEP}/merged"
    if [ ! -f "${MERGED_DIR}/config.json" ]; then
        echo "[sft175_opd300_lora32][WARN] math eval: merged missing for step=${STEP}: ${MERGED_DIR}"
        MATH_FAIL+=("${STEP}(no-merged)")
        continue
    fi
    ABBR="${MODEL_TAG_SAFE}-sft${SFT_MAX_STEPS}-lora${LORA_RANK}_opd_step${STEP}"
    STEP_EVAL_LOG="${RUN_DIR}/opd/math_eval_step${STEP}.log"
    echo "[sft175_opd300_lora32] math eval step=${STEP}: ${MERGED_DIR}"
    set +e
    MODEL_PATH="${MERGED_DIR}" \
    MODEL_TAG="${ABBR}" \
    RUN_TS="${TS}_step${STEP}" \
    EVAL_ROOT="${MATH_EVAL_ROOT}" \
        bash "${MATH_EVAL_SCRIPT}" 2>&1 | tee "${STEP_EVAL_LOG}"
    ERC=${PIPESTATUS[0]}
    set -e
    if [ "${ERC}" -ne 0 ]; then
        echo "[sft175_opd300_lora32][WARN] math eval step=${STEP} exited rc=${ERC}"
        MATH_FAIL+=("${STEP}(rc=${ERC})")
    fi
done
