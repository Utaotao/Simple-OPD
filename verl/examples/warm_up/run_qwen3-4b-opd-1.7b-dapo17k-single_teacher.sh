#!/bin/bash
# Fail-fast: any command failure (including inside the training python -m ...)
# aborts this script immediately, so an outer pipeline can rely on the exit
# code to know whether OPD really finished OK.
set -eo pipefail
set -x
export PYTHONUNBUFFERED=1

# Ensure conda env's libpython3.10.so.1.0 can be found by the dynamic linker
# (fixes: ImportError: libpython3.10.so.1.0: cannot open shared object file)
export LD_LIBRARY_PATH=/root/miniconda3/envs/verl/lib:${LD_LIBRARY_PATH:-}
export TORCH_NCCL_WATCHDOG_TIMEOUT_SEC=21600
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=21600
export NCCL_TIMEOUT=21600
# Optional: enable NCCL flight recorder so next timeout shows exact stuck rank/op
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export WANDB_API_KEY=${WANDB_API_KEY:-""}
# Allow outer pipeline to force-offline (WANDB_MODE=offline) if the cluster
# has no outbound network; default to online to keep old behaviour.
export WANDB_MODE=${WANDB_MODE:-online}
export USED_MODEL="no_api"
WANDB_PROJECT_OPD=${WANDB_PROJECT_OPD:-warm_up_opd}
# Project roots (absolute paths -- avoid HFValidationError from relative "../")
# but the outer pipeline uses the _1324356 alias, so we align on it here to
# avoid confusing mixed paths in logs.
PROJECT_ROOT=${PROJECT_ROOT:-your/absolute/path/to/project_root}
VERL_ROOT=${PROJECT_ROOT}/Simple-OPD/verl
MODEL_DIR=${PROJECT_ROOT}/models
DATA_DIR=${PROJECT_ROOT}/Simple-OPD/data
# checkpoints_dir / wandb_dir can be overridden by outer pipeline so multiple
# OPD runs (one per SFT step) do not step on each other.
checkpoints_dir=${CHECKPOINTS_DIR:-${PROJECT_ROOT}/checkpoints}
export WANDB_DIR=${WANDB_DIR:-${checkpoints_dir}/wandb_logs}
mkdir -p "${checkpoints_dir}" "${WANDB_DIR}"
# Make sure cwd is deterministic for any other relative paths inside verl
cd "${VERL_ROOT}"

AMC23_test_path=${DATA_DIR}/amc23/test.parquet

test_files="['$AMC23_test_path']"

# ---- Student / Teacher ----
# Student may be:
#   - a HF hub id (e.g. "Qwen/Qwen3-1.7B")
#   - an absolute path to an HF-format model dir (e.g. a merged SFT ckpt)
# When called from the SFT->OPD pipeline, STUDENT and STUDENT_TAG are set so
# that different SFT step ckpts (all named "global_step_100/200/...") do not
# collide on the same run_name.
Student=${STUDENT:-Qwen/Qwen3-1.7B-Base}
Teacher=${TEACHER:-RedKiKi/Qwen3-8b-base-rl-dapo-17k}
Student_tag=${STUDENT_TAG:-$(basename "${Student%/}")}
Teacher_tag=$(basename "${Teacher%/}")

batch_size=${BATCH_SIZE:-128}
lr=${OPD_LR:-1e-6}
rollout_n=${ROLLOUT_N:-1}
max_response_length=${MAX_RESPONSE_LENGTH:-8192}
eval_step=${EVAL_STEP:-75}
total_training_steps=${TOTAL_TRAINING_STEPS:-75}
save_freq=${SAVE_FREQ:-25}
# test_freq=-1 disables in-training validation; we run a full math-eval
# after training via run_eval_after_opd.sh anyway.
test_freq=${TEST_FREQ:-25}
n_gpus_per_node=${NPROC:-8}
# Allow override from outer sweep wrapper via env var ROLLOUT_TEMPERATURE
rollout_temperature=${ROLLOUT_TEMPERATURE:-1.0}

# Encode key swept hyperparams into the run name so checkpoints / wandb / eval
# outputs from different sweep points do not overwrite each other.
run_name_base="${Student_tag}OPD_Teacher_${Teacher_tag}-dapo_bs${batch_size}_lr${lr}_rolloutn${rollout_n}_rsp${max_response_length}_temp${rollout_temperature}"
# Optional suffix appended by outer pipeline (e.g. a global timestamp) to
# fully disambiguate re-runs of the same student/teacher pair.
run_name="${run_name_base}${RUN_NAME_SUFFIX:-}"

echo "[opd] Student            = ${Student}"
echo "[opd] Student_tag        = ${Student_tag}"
echo "[opd] Teacher            = ${Teacher}"
echo "[opd] run_name           = ${run_name}"
echo "[opd] checkpoints_dir    = ${checkpoints_dir}"
echo "[opd] total_train_steps  = ${total_training_steps}"
echo "[opd] save_freq          = ${save_freq}"

# Standard OPD
/root/miniconda3/envs/verl/bin/python -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        algorithm.rollout_correction.rollout_is=token \
        algorithm.rollout_correction.rollout_is_threshold=5.0 \
        algorithm.rollout_correction.rollout_rs=null \
        algorithm.rollout_correction.bypass_mode=false \
        actor_rollout_ref.rollout.calculate_log_probs=true \
        actor_rollout_ref.nccl_timeout=21600 \
        data.train_files=${DATA_DIR}/DAPO-Math-17k/data/train.parquet \
        data.val_files="$test_files" \
        data.train_batch_size=${batch_size} \
        data.max_prompt_length=2048 \
        data.max_response_length=${max_response_length} \
        data.truncation='error' \
        data.shuffle=True \
        data.seed=42 \
        data.return_raw_chat=True \
        +data.apply_chat_template_kwargs.enable_thinking=False \
        actor_rollout_ref.model.path=${Student} \
        +actor_rollout_ref.ref.model.path=${Teacher} \
        actor_rollout_ref.actor.optim.lr=${lr} \
        actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=${batch_size} \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
        actor_rollout_ref.model.enable_gradient_checkpointing=False \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
        actor_rollout_ref.rollout.n=${rollout_n} \
        actor_rollout_ref.rollout.max_num_batched_tokens=65536 \
        actor_rollout_ref.rollout.temperature=${rollout_temperature} \
        actor_rollout_ref.rollout.top_p=1.0 \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
        actor_rollout_ref.rollout.val_kwargs.top_k=20 \
        actor_rollout_ref.rollout.val_kwargs.top_p=0.8 \
        actor_rollout_ref.rollout.val_kwargs.n=1 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.ref.fsdp_config.param_offload=False \
        algorithm.use_kl_in_reward=False \
        reward_model.reward_manager=naive \
        trainer.critic_warmup=0 \
        trainer.val_before_train=True \
        trainer.logger='["console","wandb"]' \
        trainer.log_val_generations=10 \
        trainer.project_name=${WANDB_PROJECT_OPD} \
        trainer.experiment_name="${run_name}" \
        trainer.n_gpus_per_node=${n_gpus_per_node} \
        trainer.nnodes=1 \
        trainer.save_freq=${save_freq} \
        trainer.rollout_data_dir=${PROJECT_ROOT}/rollout_logs/${run_name} \
        trainer.default_local_dir=${checkpoints_dir}/${run_name} \
        trainer.test_freq=${test_freq} \
        trainer.total_training_steps=${total_training_steps} \
        trainer.total_epochs=5 $@

# ---- merge final FSDP ckpt -> HF model ----
# When SKIP_MERGE=1 (set by outer launcher), skip the built-in single-final-step
# merge entirely -- the outer launcher takes care of merging every saved shard.
# This avoids the bug where the built-in merge tries to load
# global_step_${total_training_steps}/actor even when training was interrupted
# early (e.g. wandb/ray watchdog crash), which crashes this whole script with
# set -eo pipefail and prevents the outer launcher from reaching its eval
# stages.
if [ "${SKIP_MERGE:-0}" = "1" ]; then
    echo "[opd] SKIP_MERGE=1, skipping built-in final-step merge (outer launcher will merge all saved shards)."
else
    FINAL_STEP=${FINAL_STEP:-${total_training_steps}}
    MERGED_DIR=${MERGED_DIR:-${PROJECT_ROOT}/models_warmup/${run_name}_step${FINAL_STEP}}
    FINAL_SHARD_DIR=${checkpoints_dir}/${run_name}/global_step_${FINAL_STEP}/actor
    if [ ! -d "${FINAL_SHARD_DIR}" ]; then
        echo "[opd][WARN] final shard dir does not exist, cannot merge: ${FINAL_SHARD_DIR}"
        echo "[opd][WARN] (training probably did not reach global_step_${FINAL_STEP}); skipping built-in merge."
    else
        echo "[opd] merging global_step_${FINAL_STEP} -> ${MERGED_DIR}"
        /root/miniconda3/envs/verl/bin/python -m verl.model_merger merge \
            --backend fsdp \
            --local_dir ${FINAL_SHARD_DIR} \
            --target_dir ${MERGED_DIR}
    fi
fi
#     --hf_upload_path <你的HF用户名>/Qwen3-4B-Non-Thinking-GRPO-Math-300step \
#     --private


# ---- eval merged HF ----
if [ "${SKIP_EVAL:-0}" = "1" ]; then
    echo "[opd] SKIP_EVAL=1, skipping eval"
else
    MODEL_PATH=${MERGED_DIR} \
    MODEL_TAG=${MODEL_TAG:-${run_name}_step${FINAL_STEP}} \
    RUN_TS=${RUN_TS:-$(date +%Y%m%d_%H%M%S)} \
    EVAL_ROOT=${EVAL_ROOT:-${PROJECT_ROOT}/eval_pos} \
    bash ${PROJECT_ROOT}/Simple-OPD/math_eval/run_eval_after_opd.sh 
fi
