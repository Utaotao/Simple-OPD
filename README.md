<div align="center">

# Simple-OPD: Demystifying Warm-up for On-policy Distillation

[![Paper](https://img.shields.io/badge/arXiv-2608.06802-b31b1b.svg)](https://arxiv.org/abs/2608.06802)
[![Code](https://img.shields.io/badge/Code-GitHub-black.svg)](https://github.com/Utaotao/Simple-OPD)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Tao Liu · Taiqiang Wu · Mao Zheng · Xuan Luo · Runming Yang · Xue Yang · Junjie Wang · Yujiu Yang**

</div>



## Installation

The training codes are built on [GOPD](https://github.com/RUCBM/G-OPD) which is built on [verl](https://github.com/volcengine/verl) v0.6.1 and uses vLLM for rollout generation. The provided environment installer targets Linux x86-64, Python 3.10, NVIDIA CUDA 12, and the FSDP backend. 

```bash
conda create -n verl python=3.10 -y
conda activate verl

cd verl
USE_MEGATRON=0 USE_SGLANG=0 bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .
pip install math-verify
cd ..
```


## Data Preparation

### Warm-up data

The warm-up input is a Parquet file containing at least `prompt` and `response`. The bundled launcher also uses `is_correct` when selecting a subset:

| Column | Type | Description |
|---|---|---|
| `prompt` | string or chat-message list | Prompt sampled from the downstream OPD distribution |
| `response` | string | Teacher-generated CoT followed by an explicit `Answer:` line |
| `is_correct` | boolean | Whether the response reaches the gold answer; required for `correct` or `wrong` filtering |

The launcher accepts a mixed rollout file or a pre-filtered file. `SFT_SAMPLE_MODE=correct` selects correct responses, `wrong` selects incorrect responses, and `all` keeps both. Example positive and negative files use the names `data/sft_pos.parquet` and `data/sft_neg.parquet`.


## Quick Start

[`verl/examples/warm_up/Simple-OPD.sh`](verl/examples/warm_up/Simple-OPD.sh) is the end-to-end launcher. The current script contains machine-specific paths, so first edit `VERL_CONDA_PREFIX`, `project_root`, `WANDB_API_KEY`, and `MATH_EVAL_SCRIPT` in the script for your environment. Its SFT helper uses paths relative to the warm-up directory; launch it from there:

```bash
cd verl/examples/warm_up
bash Simple-OPD.sh
```




### Pipeline outputs

One run is written below `OUTPUT_ROOT` with this layout:

```text
<model>-sft<S>-lora<R>-opd<T>_<timestamp>/
├── full_run.log
├── ckpt/global_step_<S>/                 # sharded warm-up checkpoint
├── hf/global_step_<S>/                   # merged warm-up HF model
├── opd/train/<run-name>/global_step_*/actor/
├── opd/step<k>/merged/                    # merged OPD HF checkpoints
└── opd/math_eval/                         # evaluation artifacts
```

The launcher folds the LoRA adapter into the warm-up Hugging Face checkpoint, trains OPD, merges every saved FSDP actor checkpoint, and evaluates the warm-up model plus each merged OPD checkpoint.

## Standalone Math Evaluation

The lightweight vLLM evaluator supports AIME 2024, AIME 2025, AMC 2023, and MATH-500. AIME and AMC JSONL files live under `data/`.

Evaluate one Hugging Face checkpoint:

```bash
MODEL_PATH=/path/to/hf-checkpoint \
GPUS=0,1,2,3,4,5,6,7 \
bash math_eval/start_math_eval.sh
```

Evaluate every `step*/merged` checkpoint produced by the main launcher:

```bash
OPD_ROOT=/path/to/run/opd \
GPUS=0,1,2,3,4,5,6,7 \
bash math_eval/start_math_eval.sh
```

Use `BENCHMARKS="aime24 aime25"` to evaluate a subset. Each run writes generated responses, logs, `summary.txt`, and `summary.csv` below `EVAL_ROOT`.

## Repository Structure

```text
Simple-OPD/
├── data/                           # warm-up and math-evaluation data
├── math_eval/                      # vLLM math evaluator and launchers
└── verl/
    ├── examples/warm_up/           # Simple-OPD pipeline and helpers
    └── verl/                       # training framework implementation
```

## Acknowledgments

This project is built on [verl](https://github.com/volcengine/verl) and [vLLM](https://github.com/vllm-project/vllm). We thank their developers and the maintainers of the evaluation datasets used in the paper.

## Citation

If you find Simple-OPD useful, please cite:

```bibtex
@article{simpleopd,
  title={Simple-OPD: Demystifying Warm-up for On-policy Distillation},
  author={Liu, Tao and Wu, Taiqiang and Zheng, Mao and Luo, Xuan and Yang, Runming and Yang, Xuewei and Wang, Junjie and Yang, Yujiu},
  journal={arXiv preprint arXiv:2608.06802},
  year={2026}
}
```
For any questions, feel free to pull an issue or email at `liu-t25@mails.tsinghua.edu.cn`
