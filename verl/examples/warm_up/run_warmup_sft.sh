#!/bin/bash
# ======================================================================
# Warm-up SFT on rollout-correct samples produced by
#   examples/g_opd/simple_infer_dapo17k.py
#
# LlamaFactory param -> verl param mapping:
#   --finetuning_type full        -> model.lora_rank=0 (default full-param)
#   --cutoff_len 16384            -> data.max_length=16384
#   --output_dir ...              -> trainer.default_local_dir=...
#   --per_device_train_batch_size -> data.micro_batch_size_per_gpu=1
#   --gradient_accumulation_steps -> implicit via
#                                     data.train_batch_size
#                                     = micro_bsz * n_gpus * grad_accum
#   --learning_rate 5e-6          -> optim.lr=5e-6
#   max_steps=400                 -> trainer.total_training_steps=400
#   save_steps=100                -> trainer.save_freq=100
#   --prob_threshold              -> ignored (per user request)
# ======================================================================
set -euo pipefail
set -x

# --------- required inputs ----------
# rollout parquet produced by simple_infer_dapo17k.py, must have `is_correct`.
# NOTE: this is an ABSOLUTE path and is INDEPENDENT of OUTPUT_ROOT.
#       override via `SFT_PARQUET=... bash run_warmup_sft.sh`.
: "${SFT_PARQUET:=your/absolute/path/to/rollout_sft.parquet}"

# base model to fine-tune from (same one used for rollout)
: "${MODEL_PATH:=Qwen/Qwen3-1.7B-Base}"

# output root (per-run timestamped dir goes underneath).
# One full experiment (SFT ckpts + merged HF + eval + optional downstream OPD)
# lives entirely inside ${OUTPUT_ROOT}/<MODEL_TAG>_<TS>/, so this dir is best
# thought of as an "experiment root", not just an SFT ckpt store.
#
# Layout under OUTPUT_ROOT:
#   ${OUTPUT_ROOT}/${MODEL_TAG_SAFE}_${TS}/    <- one experiment (RUN_DIR)
#     ckpt/global_step_*/                     <- SFT FSDP shards
#     hf/global_step_*/                       <- SFT merged HF ckpts
#     eval/...                                <- SFT-side math eval
#     opd/step*/...                           <- (optional) downstream OPD runs
#                                                 written by run_pipeline_sft_then_opd.sh
: "${OUTPUT_ROOT:=your/absolute/path/to/output_root}"

# --------- eval knobs ----------
# after every merged HF step, run the math-benchmark evaluator on it.
# set SKIP_EVAL=1 to disable; set EVAL_BENCHMARKS to a space/comma list
# (e.g. "aime24 amc23") to only eval a subset.
: "${EVAL_SCRIPT:=${Simple-OPD_ROOT}/math_eval/run_eval_after_opd.sh}"
: "${SKIP_EVAL:=0}"
: "${EVAL_BENCHMARKS:=}"

# sanity check: training data must exist and be readable
if [ ! -f "${SFT_PARQUET}" ]; then
    echo "[warmup_sft][ERROR] SFT_PARQUET not found: ${SFT_PARQUET}"
    echo "[warmup_sft][ERROR] please pass SFT_PARQUET=<abs-path-to-rollout-sft.parquet> when launching this script."
    echo "[warmup_sft][ERROR] example:"
    echo "    MODEL_PATH=Qwen/Qwen3-1.7B bash $0"
    exit 1
fi

# --------- hyper-params ----------
CUTOFF_LEN=${CUTOFF_LEN:-8192}
LR=${LR:-5e-6}
MICRO_BSZ=${MICRO_BSZ:-1}
GRAD_ACC=${GRAD_ACC:-1}
MAX_STEPS=${MAX_STEPS:-400}
SAVE_FREQ=${SAVE_FREQ:-100}
NPROC=${NPROC:-8}
ULYSSES_SP=${ULYSSES_SP:-1}

# --------- LoRA knobs (LORA_RANK=0 -> full-param SFT, backwards compatible) ---
#   LORA_RANK            0 => full-param (default). >0 => LoRA with this rank.
#   LORA_ALPHA           LoRA scaling factor. Defaults to LORA_RANK (so alpha/r=1).
#   LORA_TARGET_MODULES  which linear layers to attach LoRA to.
#                        Default 'all-linear' matches verl's SFT peft example.
LORA_RANK=${LORA_RANK:-0}
LORA_ALPHA=${LORA_ALPHA:-${LORA_RANK}}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-all-linear}

# global batch size = micro_bsz * n_gpus * grad_acc
GLOBAL_BSZ=$(( MICRO_BSZ * NPROC * GRAD_ACC ))

# derive a filesystem-safe short tag from MODEL_PATH:
#   "Qwen/Qwen3-1.7B"                 -> "Qwen3-1.7B"
#   "/abs/path/to/Qwen3-1.7B"         -> "Qwen3-1.7B"
#   "/abs/path/to/Qwen3-1.7B/"        -> "Qwen3-1.7B"
#
# Single-source-of-truth policy:
#   * If the caller (start.sh / run_full_pipeline.sh) already exported
#     MODEL_TAG_SAFE, reuse it verbatim.  Do NOT recompute -- historically
#     recomputation here (with `echo | tr`) produced a trailing "_" that
#     the caller's version (with `sed 's/_*$//'`) did not, making the
#     resulting RUN_DIR paths diverge (e.g. "Qwen3-1.7B-Base" vs
#     "Qwen3-1.7B-Base_") and creating two sibling dirs per run.
#   * Otherwise compute locally, using `printf` instead of `echo` so the
#     trailing newline echo appends is not turned into a "_" by `tr`.
if [ -z "${MODEL_TAG_SAFE:-}" ]; then
    MODEL_TAG_RAW="$(basename "${MODEL_PATH%/}")"
    MODEL_TAG_SAFE="$(printf '%s' "${MODEL_TAG_RAW}" | tr -c 'A-Za-z0-9._-' '_')"
fi

# resolve SFT_SAMPLE_MODE early so it can be baked into RUN_DIR name
: "${SFT_SAMPLE_MODE:=correct}"
case "${SFT_SAMPLE_MODE}" in
    correct|wrong|all) ;;
    *)
        echo "[warmup_sft][ERROR] unknown SFT_SAMPLE_MODE='${SFT_SAMPLE_MODE}' (expected: correct|wrong|all)"
        exit 1
        ;;
esac

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
# Same policy for RUN_DIR: prefer the caller-provided value (guaranteed to
# match the driver / start.sh view of the world); only fall back to a
# locally-computed one when running this script standalone.
: "${RUN_DIR:=${OUTPUT_ROOT}/${MODEL_TAG_SAFE}-${SFT_SAMPLE_MODE}_${TS}}"
mkdir -p "${RUN_DIR}"

# eval artefacts land INSIDE this run's dir, next to ckpt/ and hf/.
# structure: ${RUN_DIR}/eval/<EVAL_TAG>/all_<RUN_TS>/...
EVAL_ROOT_DIR="${RUN_DIR}/eval"
mkdir -p "${EVAL_ROOT_DIR}"

# --------- 1) filter rollout parquet ----------
# SFT_SAMPLE_MODE selects which subset of the rollout parquet is used for SFT:
#   correct (default) -> only rows with is_correct==True   (--only_correct 1)
#   wrong             -> only rows with is_correct==False  (--only_correct 0 --only_wrong 1)
#   all               -> keep all rows                     (--only_correct 0)
# NOTE: SFT_SAMPLE_MODE has already been validated above (before RUN_DIR).
case "${SFT_SAMPLE_MODE}" in
    correct)
        FILTER_ARGS="--only_correct 1"
        FILTERED_PARQUET="${RUN_DIR}/sft_correct.parquet"
        ;;
    wrong)
        FILTER_ARGS="--only_correct 0 --only_wrong 1"
        FILTERED_PARQUET="${RUN_DIR}/sft_wrong.parquet"
        ;;
    all)
        FILTER_ARGS="--only_correct 0"
        FILTERED_PARQUET="${RUN_DIR}/sft_all.parquet"
        ;;
esac
echo "[warmup_sft] SFT_SAMPLE_MODE    = ${SFT_SAMPLE_MODE}"
python ./filter_correct_sft.py \
    --input_parquet "${SFT_PARQUET}" \
    --output_parquet "${FILTERED_PARQUET}" \
    ${FILTER_ARGS}

echo "[warmup_sft] run_dir            = ${RUN_DIR}"
echo "[warmup_sft] filtered parquet   = ${FILTERED_PARQUET}"
echo "[warmup_sft] model              = ${MODEL_PATH}"
echo "[warmup_sft] cutoff_len         = ${CUTOFF_LEN}"
echo "[warmup_sft] micro_bsz/gpu      = ${MICRO_BSZ}"
echo "[warmup_sft] grad_acc           = ${GRAD_ACC}"
echo "[warmup_sft] nproc              = ${NPROC}"
echo "[warmup_sft] global_batch_size  = ${GLOBAL_BSZ}"
echo "[warmup_sft] lr                 = ${LR}"
echo "[warmup_sft] max_steps          = ${MAX_STEPS}"
echo "[warmup_sft] save_freq          = ${SAVE_FREQ}"
echo "[warmup_sft] lora_rank          = ${LORA_RANK}"
if [ "${LORA_RANK}" != "0" ]; then
    echo "[warmup_sft] lora_alpha         = ${LORA_ALPHA}"
    echo "[warmup_sft] target_modules     = ${LORA_TARGET_MODULES}"
fi

# LoRA-specific hydra args, only appended when LORA_RANK>0.
LORA_HYDRA_ARGS=""
if [ "${LORA_RANK}" != "0" ]; then
    LORA_HYDRA_ARGS="model.lora_alpha=${LORA_ALPHA} model.target_modules=${LORA_TARGET_MODULES}"
fi

# --------- 2) launch verl FSDP SFT trainer ----------
# NOTE:
#  - use val_files = train (verl requires a val_files, but with save_freq/test_freq
#    logic below we just point it at the same parquet; test_freq=-1 disables val).
#  - full-param SFT: keep model.lora_rank=0 (default).
#  - use_remove_padding + fsdp2 for memory efficiency at 16k context.
    torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC}" \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="${FILTERED_PARQUET}" \
    data.val_files="${FILTERED_PARQUET}" \
    data.val_max_samples=8 \
    data.prompt_key=prompt \
    data.response_key=response \
    data.max_length="${CUTOFF_LEN}" \
    data.truncation=right \
    data.train_batch_size="${GLOBAL_BSZ}" \
    data.micro_batch_size_per_gpu="${MICRO_BSZ}" \
    model.partial_pretrain="${MODEL_PATH}" \
    model.trust_remote_code=True \
    model.enable_gradient_checkpointing=True \
    model.strategy=fsdp2 \
    model.lora_rank="${LORA_RANK}" \
    ${LORA_HYDRA_ARGS} \
    optim.lr="${LR}" \
    optim.lr_warmup_steps_ratio=0.0 \
    optim.lr_scheduler=cosine \
    ulysses_sequence_parallel_size="${ULYSSES_SP}" \
    use_remove_padding=True \
    trainer.default_local_dir="${RUN_DIR}/ckpt" \
    trainer.project_name="${WANDB_PROJECT:-warm_up_sft}" \
    trainer.experiment_name="${MODEL_TAG_SAFE}-${SFT_SAMPLE_MODE}-${TS}" \
    trainer.logger='[console,wandb]' \
    trainer.total_epochs=100 \
    trainer.total_training_steps="${MAX_STEPS}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq=-1 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node="${NPROC}" \
    trainer.resume_mode=disable \
    trainer.device=cuda \
    2>&1 | tee -a "${RUN_DIR}/train.log"

echo "[warmup_sft] training done. sharded checkpoints saved under ${RUN_DIR}/ckpt"

# --------- 3) merge FSDP sharded ckpt -> HuggingFace model (every save step) ----------
# verl SFT saves under: ${RUN_DIR}/ckpt/global_step_${step}/
#   - model_world_size_${NPROC}_rank_*.pt   (FSDP shards)
#   - optim_world_size_${NPROC}_rank_*.pt   (optimizer shards)
#   - extra_state_world_size_${NPROC}_rank_*.pt
#   - fsdp_config.json
#   - huggingface/                          (only tokenizer + config, NO weights)
#   - data.pt
# We call `verl.model_merger` on EVERY global_step_* to produce a full HF model dir.

HF_ROOT="${RUN_DIR}/hf"
mkdir -p "${HF_ROOT}"

TRC_FLAG=""
if [ "${TRUST_REMOTE_CODE:-1}" = "1" ]; then
    TRC_FLAG="--trust-remote-code"
fi

CKPT_LIST=$(ls -1d "${RUN_DIR}"/ckpt/global_step_* 2>/dev/null | sort -V || true)
if [ -z "${CKPT_LIST}" ]; then
    echo "[warmup_sft][ERROR] no checkpoint found under ${RUN_DIR}/ckpt, cannot merge."
    exit 1
fi

# ---- optional whitelist: only merge/eval these SFT steps ----
# KEEP_STEPS is a space-separated list of integers, e.g. "100 200 300 400".
# When set, any global_step_N whose N is NOT in KEEP_STEPS is skipped
# (both merge and eval). When unset/empty, all saved steps are processed
# (backwards-compatible default).
: "${KEEP_STEPS:=}"
if [ -n "${KEEP_STEPS}" ]; then
    echo "[warmup_sft] KEEP_STEPS whitelist active: '${KEEP_STEPS}'"
fi
_step_in_keep_list() {
    # $1 = integer step. Returns 0 if KEEP_STEPS is empty or contains it.
    local n="$1"
    if [ -z "${KEEP_STEPS}" ]; then return 0; fi
    for k in ${KEEP_STEPS}; do
        if [ "${k}" = "${n}" ]; then return 0; fi
    done
    return 1
}

LAST_HF_DIR=""
for SHARDED_CKPT_DIR in ${CKPT_LIST}; do
    STEP_NAME=$(basename "${SHARDED_CKPT_DIR}")   # e.g. global_step_100
    STEP_NUM="${STEP_NAME#global_step_}"          # e.g. 100
    MERGED_HF_DIR="${HF_ROOT}/${STEP_NAME}"

    if ! _step_in_keep_list "${STEP_NUM}"; then
        echo "[warmup_sft] KEEP_STEPS filter: skip ${STEP_NAME} (not in whitelist)"
        continue
    fi

    if [ -f "${MERGED_HF_DIR}/config.json" ] && \
       ( ls "${MERGED_HF_DIR}"/*.safetensors >/dev/null 2>&1 || ls "${MERGED_HF_DIR}"/pytorch_model*.bin >/dev/null 2>&1 ); then
        echo "[warmup_sft] skip merge (already exists): ${MERGED_HF_DIR}"
    else
        echo "[warmup_sft] merging ${STEP_NAME} -> ${MERGED_HF_DIR}"
        mkdir -p "${MERGED_HF_DIR}"
        python -m verl.model_merger merge \
            --backend fsdp \
            --local_dir "${SHARDED_CKPT_DIR}" \
            --target_dir "${MERGED_HF_DIR}" \
            ${TRC_FLAG} \
            2>&1 | tee -a "${RUN_DIR}/merge.log"
    fi

    # ---- LoRA post-merge fix -------------------------------------------------
    # verl.model_merger does NOT actually merge LoRA weights into the base
    # tensors -- it only splits them out into `lora_adapter/` and writes the
    # RAW BASE weights to model.safetensors (see save_lora_adapter() in
    # verl/model_merger/base_model_merger.py). It also hardcodes
    # `lora_alpha=0` in adapter_config.json (with a "should be set manually"
    # comment). As a result, without this extra step the top-level HF
    # checkpoint is identical to the pretrained base model, and downstream
    # eval / OPD-student loading would silently use the un-adapted base.
    #
    # We therefore run a dedicated merger that:
    #   1) fixes lora_alpha in adapter_config.json (defaults to r, matching
    #      this repo's `LORA_ALPHA=${LORA_RANK}` convention),
    #   2) uses PEFT `merge_and_unload()` to fold the adapter into the base,
    #   3) overwrites model.safetensors in place with the merged weights.
    #
    # No-op for full-param SFT runs (no lora_adapter/ present).
    if [ -d "${MERGED_HF_DIR}/lora_adapter" ] && [ ! -f "${MERGED_HF_DIR}/.lora_merged" ]; then
        LORA_MERGE_LOG="${RUN_DIR}/lora_merge_${STEP_NAME}.log"
        echo "[warmup_sft] detected LoRA adapter under ${MERGED_HF_DIR}/lora_adapter"
        echo "[warmup_sft]   -> folding it into base weights (log: ${LORA_MERGE_LOG})"
        # LORA_ALPHA / MODEL_PATH are already exported / defined above.
        # Fall back to the value stored in adapter_config.json (r) if
        # LORA_ALPHA is unset or 0 for some reason.
        _EFFECTIVE_LORA_ALPHA="${LORA_ALPHA:-0}"
        _LORA_ALPHA_ARG=""
        if [ "${_EFFECTIVE_LORA_ALPHA}" != "0" ]; then
            _LORA_ALPHA_ARG="--lora-alpha ${_EFFECTIVE_LORA_ALPHA}"
        fi
        set +e
        python ./merge_lora_into_base.py \
            --hf-dir "${MERGED_HF_DIR}" \
            ${_LORA_ALPHA_ARG} \
            --base-model-name "${MODEL_PATH}" \
            --dtype bf16 \
            2>&1 | tee "${LORA_MERGE_LOG}"
        LORA_MERGE_RC=${PIPESTATUS[0]}
        set -e
        if [ "${LORA_MERGE_RC}" -ne 0 ]; then
            echo "[warmup_sft][ERROR] LoRA post-merge failed for ${STEP_NAME}, rc=${LORA_MERGE_RC}"
            echo "[warmup_sft][ERROR] see: ${LORA_MERGE_LOG}"
            exit "${LORA_MERGE_RC}"
        fi
    elif [ -f "${MERGED_HF_DIR}/.lora_merged" ]; then
        echo "[warmup_sft] LoRA already folded into base for ${STEP_NAME} (marker .lora_merged present), skipping"
    fi
    # -------------------------------------------------------------------------

    LAST_HF_DIR="${MERGED_HF_DIR}"

    # ---- eval this merged step on math benchmarks ----
    if [ "${SKIP_EVAL}" = "1" ]; then
        echo "[warmup_sft] SKIP_EVAL=1, skipping eval for ${STEP_NAME}"
    elif [ ! -f "${EVAL_SCRIPT}" ]; then
        echo "[warmup_sft][WARN] EVAL_SCRIPT not found: ${EVAL_SCRIPT} ; skipping eval."
    else
        EVAL_TAG="warmup_sft_${MODEL_TAG_SAFE}_${TS}_${STEP_NAME}"   # e.g. warmup_sft_Qwen3-1.7B_20260702_223015_global_step_100
        EVAL_LOG="${RUN_DIR}/eval_${STEP_NAME}.log"
        echo "[warmup_sft] running math eval on ${STEP_NAME}"
        echo "[warmup_sft]   MODEL_PATH = ${MERGED_HF_DIR}"
        echo "[warmup_sft]   MODEL_TAG  = ${EVAL_TAG}"
        echo "[warmup_sft]   EVAL_ROOT  = ${EVAL_ROOT_DIR}"
        echo "[warmup_sft]   log        = ${EVAL_LOG}"
        set +e
        MODEL_PATH="${MERGED_HF_DIR}" \
        MODEL_TAG="${EVAL_TAG}" \
        RUN_TS="${TS}_${STEP_NAME}" \
        EVAL_ROOT="${EVAL_ROOT_DIR}" \
            bash "${EVAL_SCRIPT}" ${EVAL_BENCHMARKS} \
            2>&1 | tee "${EVAL_LOG}"
        EVAL_RC=${PIPESTATUS[0]}
        set -e
        if [ "${EVAL_RC}" -ne 0 ]; then
            echo "[warmup_sft][WARN] eval for ${STEP_NAME} exited with rc=${EVAL_RC}; continuing to next step."
        else
            echo "[warmup_sft] eval for ${STEP_NAME} finished ok."
        fi
    fi
done

# --------- 4) summary ----------
echo ""
echo "=================== warm-up SFT paths ==================="
echo "output root         : ${OUTPUT_ROOT}"
echo "run dir             : ${RUN_DIR}"
echo "  train log         : ${RUN_DIR}/train.log"
echo "  merge log         : ${RUN_DIR}/merge.log"
echo "  filtered parquet  : ${FILTERED_PARQUET}"
echo "  sharded ckpt root : ${RUN_DIR}/ckpt         (per-step FSDP shards)"
echo "  merged HF root    : ${HF_ROOT}              (per-step HuggingFace models)"
echo "    per-step dirs   :"
for SHARDED_CKPT_DIR in ${CKPT_LIST}; do
    STEP_NAME=$(basename "${SHARDED_CKPT_DIR}")
    STEP_NUM="${STEP_NAME#global_step_}"
    if ! _step_in_keep_list "${STEP_NUM}"; then continue; fi
    echo "      ${HF_ROOT}/${STEP_NAME}"
done
echo "  final HF (last step) : ${LAST_HF_DIR}"
if [ "${SKIP_EVAL}" != "1" ]; then
    echo "  eval logs (per step) : ${RUN_DIR}/eval_global_step_*.log"
    echo "  eval artefacts root  : ${EVAL_ROOT_DIR}/warmup_sft_${MODEL_TAG_SAFE}_${TS}_global_step_<N>/all_${TS}_global_step_<N>/"
fi
echo "========================================================="
