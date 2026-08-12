#!/usr/bin/env python
"""
Merge a LoRA adapter into the base weights saved by ``verl.model_merger``.

Background
==========
``verl.model_merger`` (see ``verl/model_merger/base_model_merger.py``) does NOT
actually merge LoRA weights into the base tensors when it processes a
LoRA-trained checkpoint. It only:

  1. Splits ``lora_A`` / ``lora_B`` tensors out of the FSDP state dict into a
     stand-alone ``lora_adapter/`` subfolder (peft-compatible format), and
  2. Renames ``base_model.model.xxx.base_layer.weight`` -> ``xxx.weight`` and
     saves those raw base tensors to the top-level ``model.safetensors``.

Consequence: the top-level ``model.safetensors`` is *identical to the pretrained
base model* -- downstream evaluation / OPD-student loading would silently use
the un-adapted base weights.

Worse, the adapter_config.json written by verl hardcodes ``lora_alpha=0``,
which would zero-out the LoRA scaling factor even if one tried to load the
adapter separately.

This script fixes both problems:
  * Patch ``lora_adapter/adapter_config.json`` so ``lora_alpha`` reflects the
    actual training-time value (taken from ``--lora-alpha`` or, if omitted,
    inferred to be equal to ``r``).
  * Load ``base + adapter`` with PEFT, call ``merge_and_unload()``, and
    overwrite the top-level HuggingFace weights (``model.safetensors`` +
    sharded files if any) with the properly merged tensors.

Usage
=====
    python merge_lora_into_base.py \
        --hf-dir /path/to/hf/global_step_N \
        --lora-alpha 64                      # optional; defaults to r

After running, ``<hf-dir>/model.safetensors`` (and companion shards) will hold
``W_base + (alpha/r) * B @ A`` for every LoRA target module, so any downstream
consumer of ``<hf-dir>`` will now correctly see the fine-tuned model.
The ``lora_adapter/`` sub-folder is kept as-is (for auditing / re-use).

Design notes
============
We deliberately avoid ``PeftModel.from_pretrained`` because in some
peft/safetensors versions it hard-codes ``device=torch_device`` inside
``load_peft_weights``, which triggers a cryptic
    ValueError: could not determine the shape of object type 'torch.storage.UntypedStorage'
in CPU-only environments (e.g. when running the merge in parallel with an
on-GPU eval that has taken all visible GPUs).

Instead we:
  1) build the ``PeftModel`` skeleton with ``get_peft_model(base, LoraConfig)``
     so all lora_A / lora_B parameters are randomly initialized on CPU, then
  2) load the adapter weights from ``adapter_model.safetensors`` explicitly
     with ``device='cpu'`` and copy them into the model via
     ``set_peft_model_state_dict``.

This CPU-safe path also lets the script run on a box where GPUs are busy.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _log(msg: str) -> None:
    print(f"[merge_lora] {msg}", flush=True)


def _find_adapter_dir(hf_dir: Path) -> Path | None:
    cand = hf_dir / "lora_adapter"
    if cand.is_dir() and (cand / "adapter_config.json").is_file():
        return cand
    return None


def _portable_model_reference(model_name: str | None) -> str | None:
    """Keep Hub IDs intact while stripping local directory prefixes."""
    if not model_name:
        return None
    candidate = Path(model_name).expanduser()
    looks_local = (
        candidate.is_absolute()
        or model_name.startswith(("./", "../", "~/"))
        or candidate.exists()
    )
    return candidate.name if looks_local else model_name


def _patch_adapter_config(adapter_dir: Path, lora_alpha: int | None,
                          base_model_name: str | None) -> dict:
    cfg_path = adapter_dir / "adapter_config.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    r = int(cfg.get("r", 0))
    if r <= 0:
        raise RuntimeError(
            f"adapter_config.json at {cfg_path} has invalid r={r}; "
            "cannot proceed."
        )

    old_alpha = cfg.get("lora_alpha", None)
    if lora_alpha is None:
        # Default policy in this repo: alpha == r (see
        # start_paired_correct_lora_sweep.sh: `LORA_ALPHA=${LORA_RANK}`).
        lora_alpha = r
    cfg["lora_alpha"] = int(lora_alpha)

    portable_base_model_name = _portable_model_reference(
        base_model_name or cfg.get("base_model_name_or_path")
    )
    if portable_base_model_name:
        cfg["base_model_name_or_path"] = portable_base_model_name

    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=4)

    _log(
        f"patched adapter_config.json: r={r}, lora_alpha "
        f"{old_alpha} -> {cfg['lora_alpha']}, "
        f"base_model_name_or_path={cfg.get('base_model_name_or_path')}"
    )
    return cfg


def _wipe_old_weight_files(hf_dir: Path) -> None:
    """Remove any pre-existing weight files under ``hf_dir`` so that a fresh
    ``save_pretrained`` does not leave stale shards next to the new ones.

    Only weight files are removed; tokenizer / config / generation_config /
    the ``lora_adapter`` sub-folder are preserved.
    """
    patterns = [
        "model.safetensors",
        "model-*.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model-*.bin",
        "pytorch_model.bin.index.json",
    ]
    for pat in patterns:
        for p in hf_dir.glob(pat):
            _log(f"removing stale weight file: {p.name}")
            p.unlink()


def merge_lora_in_place(hf_dir: Path, lora_alpha: int | None,
                        base_model_name: str | None,
                        dtype: str,
                        device: str) -> None:
    if not hf_dir.is_dir():
        raise FileNotFoundError(f"hf_dir does not exist: {hf_dir}")

    adapter_dir = _find_adapter_dir(hf_dir)
    if adapter_dir is None:
        _log(
            f"no lora_adapter/ under {hf_dir}, nothing to merge. "
            "(this is expected for full-param SFT runs.)"
        )
        return

    _log(f"hf_dir       = {hf_dir}")
    _log(f"adapter_dir  = {adapter_dir}")
    _log(f"device       = {device}")

    # 1) Patch adapter_config.json (fix lora_alpha=0, set base_model path).
    cfg = _patch_adapter_config(adapter_dir, lora_alpha, base_model_name)

    # 2) Import heavy deps only when we actually have work to do.
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict

    torch_dtype = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }.get(dtype.lower())
    if torch_dtype is None:
        raise ValueError(f"unknown --dtype: {dtype}")

    # 3) Load base weights (which is what verl currently wrote to hf_dir).
    _log(f"loading base model from {hf_dir} (dtype={dtype})")
    base = AutoModelForCausalLM.from_pretrained(
        str(hf_dir),
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    base = base.to(device)

    # 4) Attach a fresh LoRA skeleton with the SAME hyper-params as training.
    #    We construct LoraConfig from the patched adapter_config.json so that
    #    r / lora_alpha / target_modules / etc. are exactly right.
    _log("attaching a fresh LoRA skeleton (get_peft_model)")
    lora_kwargs = dict(
        r=int(cfg["r"]),
        lora_alpha=int(cfg["lora_alpha"]),
        target_modules=list(cfg["target_modules"]),
        lora_dropout=float(cfg.get("lora_dropout", 0.0) or 0.0),
        bias=str(cfg.get("bias", "none")),
        fan_in_fan_out=bool(cfg.get("fan_in_fan_out", False)),
    )
    # `task_type` is optional and often stored as None in verl-produced configs.
    if cfg.get("task_type"):
        lora_kwargs["task_type"] = cfg["task_type"]
    lora_cfg = LoraConfig(**lora_kwargs)
    model = get_peft_model(base, lora_cfg)

    # 5) Load adapter weights from disk (CPU) and copy them in.
    adapter_file = adapter_dir / "adapter_model.safetensors"
    _log(f"loading adapter tensors from {adapter_file} (device=cpu)")
    adapter_sd = load_file(str(adapter_file), device="cpu")
    _log(f"adapter state_dict: {len(adapter_sd)} tensors")

    # If we asked the model to live on a non-cpu device, move tensors there.
    if device != "cpu":
        adapter_sd = {k: v.to(device) for k, v in adapter_sd.items()}

    # Cast adapter tensors to the same dtype as the base model, so that
    # `merge_and_unload` does not silently upcast/downcast under the hood.
    adapter_sd = {k: v.to(torch_dtype) for k, v in adapter_sd.items()}

    load_result = set_peft_model_state_dict(model, adapter_sd)
    missing = getattr(load_result, "missing_keys", None) or []
    unexpected = getattr(load_result, "unexpected_keys", None) or []
    _log(f"set_peft_model_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    if unexpected:
        _log(f"  first unexpected keys: {unexpected[:3]}")
    # A perfectly-matched set should have `missing_keys` containing ONLY the
    # base_model.* frozen weights (which are legitimately not in the adapter);
    # `unexpected_keys` must be empty. We do not fail on `missing` because peft
    # spams it with all frozen base params.
    if unexpected:
        raise RuntimeError(
            f"unexpected keys when loading adapter into peft model: {unexpected[:5]}"
        )

    # 6) merge_and_unload -> gives us a plain HF model with LoRA folded in.
    _log("calling merge_and_unload() -- this may take a moment")
    merged = model.merge_and_unload()

    # Loading from the local HF directory makes Transformers store that absolute
    # directory in config._name_or_path. Replace it before saving so published
    # checkpoints do not expose a cluster mount, username, or project root.
    merged.config._name_or_path = cfg.get("base_model_name_or_path", "")

    # 7) Overwrite the HF weight files in place.
    _wipe_old_weight_files(hf_dir)
    _log(f"saving merged model to {hf_dir}")
    merged.save_pretrained(str(hf_dir), safe_serialization=True)

    # Drop a small breadcrumb so it is obvious the merge has happened.
    marker = hf_dir / ".lora_merged"
    marker.write_text(
        json.dumps(
            {
                "merged": True,
                "adapter_dir": adapter_dir.name,
                "dtype": dtype,
                "device": device,
                "r": int(cfg["r"]),
                "lora_alpha": int(cfg["lora_alpha"]),
            },
            indent=2,
        )
    )
    _log(f"done. wrote marker: {marker}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hf-dir",
        required=True,
        type=Path,
        help="HuggingFace-format checkpoint dir produced by verl.model_merger; "
             "must contain a `lora_adapter/` sub-folder if LoRA was used.",
    )
    parser.add_argument(
        "--lora-alpha",
        default=None,
        type=int,
        help="Training-time LoRA alpha. Default: equal to r "
             "(matches this repo's `LORA_ALPHA=${LORA_RANK}` convention).",
    )
    parser.add_argument(
        "--base-model-name",
        default=None,
        type=str,
        help="Value to write into adapter_config.json's "
             "`base_model_name_or_path`. Cosmetic; only affects auditing.",
    )
    parser.add_argument(
        "--dtype",
        default="bf16",
        type=str,
        help="Torch dtype used for loading & saving (bf16 / fp16 / fp32).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="Device to run the merge on. Default 'cpu' so this script can "
             "run in parallel with a GPU-bound eval; pass 'cuda' if the GPUs "
             "are free and you want a faster merge.",
    )
    args = parser.parse_args()

    try:
        merge_lora_in_place(
            hf_dir=args.hf_dir.resolve(),
            lora_alpha=args.lora_alpha,
            base_model_name=args.base_model_name,
            dtype=args.dtype,
            device=args.device,
        )
    except Exception as e:  # noqa: BLE001
        _log(f"FAILED: {e!r}")
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
