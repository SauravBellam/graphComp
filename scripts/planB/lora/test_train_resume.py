#!/usr/bin/env python3
"""Functional test of train_lora.py's session features on a tiny random-weight model with the 31B's architecture (Oct 18).

    python test_train_resume.py --data TRAJS.jsonl --work DIR

Builds a 4-layer Gemma 4 model from google/gemma-4-31b-it-qat-q4_0-unquantized's config (same chat template, tokenizer,
layer types including attention_k_eq_v full-attention layers; random weights, ~100 MB), then runs train_lora.py like
notebook 21 does (4-bit, --device-map auto, --keep-order, --tail-n, --lr-floor, --stop-at, --save-every-min):
  A. fresh LoRA on the first 200 trajectories with 20 s of training (cannot fit them all): must stop early ("deadline"), jump to
     the 3-trajectory tail, save an adapter whose trained_ids.json contains every tail trajectory, and checkpoint on the way;
  B. --init-adapter A's adapter on the same file: must print "resumed from adapter" and save again.
Exit 0 only if every check passes.
"""
import argparse, json, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = "google/gemma-4-31b-it-qat-q4_0-unquantized"


def tiny_model(dst):
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    if (dst / "config.json").exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json",
              "generation_config.json", "processor_config.json"):
        try:
            hf_hub_download(SRC, f, local_dir=dst / "src")
        except Exception as e:
            print(f"(optional {f}: {type(e).__name__})")
    cfg = AutoConfig.from_pretrained(dst / "src")
    tc = cfg.get_text_config()
    lt = list(getattr(tc, "layer_types", []) or [])
    full = next(i for i, t in enumerate(lt) if t == "full_attention")
    keep = [0, 1, 2, full]                           # three sliding layers + one full-attention (K = V) layer
    tc.layer_types = [lt[i] for i in keep]
    tc.num_hidden_layers = len(keep)
    for k, v in dict(hidden_size=256, intermediate_size=512, num_attention_heads=4, num_key_value_heads=2, head_dim=64).items():
        if hasattr(tc, k):
            setattr(tc, k, v)
    for k in ("global_head_dim", "num_global_key_value_heads"):
        if hasattr(tc, k):
            setattr(tc, k, 64 if "dim" in k else 2)
    vc = getattr(cfg, "vision_config", None)
    if vc is not None:
        for k, v in dict(hidden_size=64, intermediate_size=128, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                         head_dim=32).items():
            if hasattr(vc, k):
                setattr(vc, k, v)
    model = AutoModelForCausalLM.from_config(cfg)
    model.save_pretrained(dst)
    AutoTokenizer.from_pretrained(dst / "src").save_pretrained(dst)
    for f in ("chat_template.jinja",):
        if (dst / "src" / f).exists():
            (dst / f).write_text((dst / "src" / f).read_text())
    print(f"tiny model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M params, layers {tc.layer_types}")


def run(args, log):
    p = subprocess.run([sys.executable, str(HERE / "train_lora.py")] + args, capture_output=True, text=True)
    Path(log).write_text(p.stdout + p.stderr)
    print(f"--- {log}: exit {p.returncode}\n" + (p.stdout + p.stderr)[-2500:])
    return p.returncode, p.stdout + p.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--work", required=True)
    a = ap.parse_args()
    w = Path(a.work); w.mkdir(parents=True, exist_ok=True)
    tiny = w / "tiny31"
    tiny_model(tiny)
    rows = []
    with open(a.data) as fh:
        for l in fh:
            if l.strip():
                rows.append(json.loads(l))
            if len(rows) >= 200:
                break
    for k, r in enumerate(rows):
        r.setdefault("source_traj_id", f"test:{r['instance_id']}:{k}")
    data = w / "train.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in rows))
    tail_ids = {r["source_traj_id"] for r in rows[-3:]}
    common = ["--data", str(data), "--base", str(tiny), "--device-map", "auto", "--max-len", "32768", "--max-target", "8000",
              "--val-frac", "0.02", "--no-eval-start", "--served-model", "google/gemma-4-31b-it-qat-w4a16-ct", "--epochs", "1",
              "--keep-order", "--tail-n", "3", "--lr-floor", "0.3", "--grad-accum", "1"]
    fails = []
    # A: 20 s of training cannot fit 200 trajectories: must jump to the tail, stop on the deadline; checkpoints every ~3 s
    rc, out = run(common + ["--out", str(w / "A"), "--stop-after-s", "20", "--save-every-min", "0.05"], w / "A.log")
    ad = w / "A" / "adapter"
    trained = set(json.loads((ad / "trained_ids.json").read_text())) if (ad / "trained_ids.json").exists() else set()
    meta = json.loads((w / "A" / "train_meta.json").read_text()) if (w / "A" / "train_meta.json").exists() else {}
    checks = {"A exit 0": rc == 0, "A adapter saved": (ad / "adapter_config.json").exists(),
              "A every tail trajectory trained": tail_ids <= trained,
              "A served model in adapter_config": "w4a16" in (ad / "adapter_config.json").read_text() if ad.exists() else False,
              "A tok_per_s logged": bool(meta.get("tok_per_s")),
              "A stopped on the deadline": meta.get("stopped") == "deadline", "A not everything trained": len(trained) < len(rows)}
    jumped = "skipping" in out and "tail samples" in out
    checks["A jumped to the tail before the deadline"] = jumped
    print(f"A: stopped={meta.get('stopped')} jumped_to_tail={jumped} trained={len(trained)}/{len(rows)} tok/s={meta.get('tok_per_s')}")
    # B: resume from A's adapter
    rc, out = run(common + ["--out", str(w / "B"), "--init-adapter", str(ad), "--save-every-min", "30"], w / "B.log")
    checks.update({"B exit 0": rc == 0, "B resumed": "resumed from adapter" in out,
                   "B adapter saved": (w / "B" / "adapter" / "adapter_config.json").exists()})
    for k, v in checks.items():
        print(("PASS " if v else "FAIL ") + k)
        if not v:
            fails.append(k)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
