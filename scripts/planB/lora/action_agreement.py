#!/usr/bin/env python3
"""Next-action agreement with the 31B teacher on held-out turns (Oct 15): does a student pick the teacher's next tool call?

    python action_agreement.py --data train.jsonl --val-from <train_meta.json> --out OUT.json \
        --model e4b=<base dir> [--model e4b_rft=<base dir>+<adapter dir>] [--model e2b=<base dir>]

For every assistant turn of the val tasks (the split recorded in --val-from, so the adapter never trained on them), up to
the first compaction, the prompt is rendered exactly as train_lora.py renders it; each model generates greedily (thinking on,
as deployed) and the first tool call is compared with the teacher's: tool name, and the full call (name + arguments).
Models load as in training (nf4, bf16 compute). A cheap proxy that, unlike val loss, catches a no-op adapter.
"""
import argparse, json, re, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_lora as T  # noqa: E402  (render, turn_samples, tool schemas: identical to training)

CALL = re.compile(r"<\|tool_call>call:(\w+)(\{.*?\})?<tool_call\|>", re.S)


def first_call(text):
    m = CALL.search(text)
    return (m.group(1), m.group(0)) if m else ("<none>", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--val-from", required=True, help="train_meta.json of the adapter (its val_tasks)")
    ap.add_argument("--model", action="append", required=True, help="NAME=BASE[+ADAPTER]")
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    val = set(json.load(open(a.val_from))["val_tasks"])
    trajs = [json.loads(l) for l in open(a.data) if l.strip() and json.loads(l)["instance_id"] in val]
    tools = T.harness_tool_schemas(T.B2_TOOLS)
    res = {"val_tasks": sorted(val), "models": {}}
    for spec in a.model:
        name, path = spec.split("=", 1)
        base, _, adapter = path.partition("+")
        tok = AutoTokenizer.from_pretrained(base)
        stats = {"prefix_mismatch": 0, "mismatch_examples": [], "prompt_tok_pairs": [], "after_compaction": 0,
                 "compacted_trajectories": 0, "masked_turns": 0, "too_long": 0, "target_too_long": 0}
        # v1 returns (prompt, target), v2 (prompt, target, turn, masked)
        turns = [(t["instance_id"], s[0], s[1]) for t in trajs for s in T.turn_samples(tok, t, tools, stats)]
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                               bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map={"": 0},
                                                     attn_implementation="sdpa", quantization_config=q)
        if adapter:
            model = PeftModel.from_pretrained(model, adapter)
        model.eval()
        rows, t0 = [], time.time()
        for i, (iid, prompt, target) in enumerate(turns):
            ids = tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(0)
            with torch.no_grad():
                g = model.generate(ids, max_new_tokens=a.max_new, do_sample=False, stop_strings=["<tool_call|>"],
                                   tokenizer=tok)
            gen = tok.decode(g[0, ids.shape[1]:], skip_special_tokens=False)
            (tn, tc), (gn, gc) = first_call(target), first_call(gen)
            rows.append({"task": iid, "teacher": tn, "student": gn, "name_match": tn == gn, "call_match": tc == gc and tc != "",
                         "gen_tokens": int(g.shape[1] - ids.shape[1])})
            print(f"[{name}] {i + 1}/{len(turns)} {iid} teacher={tn} student={gn} ({time.time() - t0:.0f}s)", flush=True)
        n = len(rows)
        res["models"][name] = {"spec": path, "turns": n, "prefix_mismatch": stats["prefix_mismatch"],
                               "name_match": sum(r["name_match"] for r in rows), "call_match": sum(r["call_match"] for r in rows),
                               "no_call": sum(r["student"] == "<none>" for r in rows), "seconds": round(time.time() - t0),
                               "rows": rows}
        m = res["models"][name]
        print(f"RESULT {name}: tool name {m['name_match']}/{n}, exact call {m['call_match']}/{n}, no call {m['no_call']}/{n}", flush=True)
        del model
        torch.cuda.empty_cache()
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
