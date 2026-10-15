#!/usr/bin/env python3
"""Fold a LoRA into its base checkpoint's safetensors so PyPI vLLM (no Gemma 4 LoRA support) can serve it as a plain model.

    python merge_lora.py --base <unquantized QAT dir> --adapter <dir with adapter_config.json> --out <dir>
    python merge_lora.py --base <dir> --out <dir>          # no adapter: a servable copy of the base (symlinks)

Tensor math only (W += B @ A * alpha / r), so the config, architecture and every non-LoRA tensor stay byte-identical to the
base. Stops if any adapter tensor has no matching base weight: a partial merge would be a silent no-op, as in vLLM.
Files the base snapshot lacks but vLLM wants (processor_config.json) are copied from --extra-files-from.
"""
import argparse, json, re, shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra-files-from", default="~/gc/data/model_e4b")
    a = ap.parse_args()
    base, out = Path(a.base).expanduser(), Path(a.out).expanduser()
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    shards = sorted(base.glob("*.safetensors"))
    for f in base.iterdir():
        if f.suffix != ".safetensors" or not a.adapter:
            (out / f.name).symlink_to(f.resolve())
    for f in Path(a.extra_files_from).expanduser().glob("processor_config.json"):
        if not (out / f.name).exists():
            shutil.copy(f, out / f.name)
    if not a.adapter:
        print(f"base copy (symlinks): {out}")
        return
    ad = Path(a.adapter).expanduser()
    cfg = json.loads((ad / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    lora = load_file(ad / "adapter_model.safetensors")
    # base_model.model.<module>.lora_A.weight -> <module>.weight
    pairs = {}
    for k in lora:
        m = re.fullmatch(r"base_model\.model\.(.+)\.lora_([AB])\.weight", k)
        if not m:
            raise SystemExit(f"unexpected adapter tensor {k}")
        pairs.setdefault(m.group(1) + ".weight", {})[m.group(2)] = lora[k]
    left = set(pairs)
    for sh in shards:
        t = load_file(sh)
        for name in left & set(t):
            w = t[name]
            delta = (pairs[name]["B"].float() @ pairs[name]["A"].float()) * scale
            t[name] = (w.float() + delta).to(w.dtype)
            left.discard(name)
        save_file(t, out / sh.name, metadata={"format": "pt"})
        del t
    if left:
        shutil.rmtree(out)
        raise SystemExit(f"{len(left)} adapter modules have no base weight (e.g. {sorted(left)[:3]}): not merged")
    print(f"merged {len(pairs)} modules (scale {scale}) into {out}")


if __name__ == "__main__":
    main()
