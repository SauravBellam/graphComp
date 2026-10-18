#!/usr/bin/env python3
"""Train a LoRA on RFT / distillation trajectories (build_rft_data.py output). Plan C (Oct 14; v2 Oct 15).

    # CPU check: renders every sample with the model's chat template and the harness's own tool schemas, verifies that each
    # target turn continues its prompt, and reports token counts.
    python train_lora.py --data train.jsonl --base <unquantized QAT dir> --out OUT --check
    # E4B, every GPU data-parallel (one QLoRA copy per GPU):
    torchrun --nproc_per_node 4 train_lora.py --data ... --base <e4b dir> --out OUT
    # 31B, one process, the 4-bit model split over every GPU (layer by layer):
    python train_lora.py --data ... --base <31b dir> --out OUT --device-map auto --served-model google/gemma-4-31b-it-qat-w4a16-ct

Base: the UNQUANTIZED QAT checkpoint the served w4a16 model was made from; vLLM serves the adapter on w4a16.

Samples (v2): a trajectory is cut into WINDOWS of consecutive assistant turns whose target tokens total at most --max-target.
A window is one sequence = the conversation rendered up to its last turn; the loss is on the window's assistant turns only,
every earlier turn being context. This is the usual multi-turn SFT objective (loss on assistant tokens, full context) and costs
one forward per window instead of one per turn. A window is used only if every turn in it is an exact substring continuation of
the rendering (checked turn by turn, as v1 did); otherwise its turns fall back to one sample per turn.
Practices built in (sources in the notebook that runs this, section "Training choices"):
- assistant-only loss, full multi-turn context; train/val split by task;
- step-level error masking: an assistant turn whose tool result is an error (except run_command, whose failures are test or
  reproduction feedback) stays in the context but gets no loss (--no-error-mask turns it off);
- train = deploy: --system-prompt replaces the recorded system prompt and --deploy-budget rewrites the harness's budget lines in
  the task prompt, so the adapter is trained on exactly the text it will be served with;
- vanilla LoRA on every language-model projection, rank 32 / alpha 64, lr 1e-4, cosine with warmup, at most 2 epochs;
- 31B (attention_k_eq_v): no LoRA on k_proj of the full-attention layers (vLLM would not apply the delta to V there);
- --time-budget-h: after the first optimizer step the run shrinks its total step count to what fits, so the LR schedule
  still anneals to 0 and the adapter is always saved.
Several GPUs: torchrun -> data-parallel (ranks load one after another; LoRA gradients averaged before each step); rank 0 logs
and saves. More ranks than GPUs (smoke test only) share GPUs over gloo.
"""
import argparse, json, math, os, random, re, struct, sys, time
WORLD, RANK, LOCAL = (int(os.environ.get(k, d)) for k, d in (("WORLD_SIZE", "1"), ("RANK", "0"), ("LOCAL_RANK", "0")))
from pathlib import Path

B2_TOOLS = ["run_command", "read_file", "edit_file", "write_file", "get_status", "submit_patch",
            "get_code_neighbors", "search_similar_code", "get_code_subgraph"]


def harness_tool_schemas(names):
    """The tool list exactly as the harness sends it: swegemma tools -> ADK FunctionTool declarations -> litellm params."""
    from swegemma.tools import create_tools
    from google.adk.tools.function_tool import FunctionTool
    from google.adk.models.lite_llm import _function_declaration_to_tool_param

    class _Ctx:                          # the tool factories only bind ctx; nothing is executed here
        def __getattr__(self, k):
            return None
    tools = create_tools(_Ctx())
    return [_function_declaration_to_tool_param(FunctionTool(tools[n])._get_declaration()) for n in names]


PAST_REASONING = "keep"       # set by --past-reasoning; "drop" removes reasoning from all but the target turn
# ADK's identity processor appends this to the agent instruction on every request (google/adk/flows/llm_flows/identity.py)
SYSTEM_SUFFIX = '\n\nYou are an agent. Your internal name is "swe_planb_agent".'
SYSTEM_PROMPT = None          # --system-prompt
DEPLOY_BUDGET = None          # --deploy-budget
# the harness's budget lines in the task prompt (swegemma/harness/agent_runner.py, build prompt)
BUDGET_RE = {"time_minutes": (r"- Time allowance: [\d.]+ minutes", "- Time allowance: {} minutes"),
             "tool_calls": (r"- Tool calls allowance: \d+ calls", "- Tool calls allowance: {} calls"),
             "turns": (r"- Max loop iterations: \d+ turns", "- Max loop iterations: {} turns"),
             "timeout_seconds": (r"- Single command timeout: \d+ seconds", "- Single command timeout: {} seconds")}


def clean(msgs, target_idx=None):
    out = []
    for i, m in enumerate(msgs):
        m = {k: v for k, v in m.items() if not k.startswith("_")}
        if i == 0 and m["role"] == "system":
            m["content"] = (SYSTEM_PROMPT if SYSTEM_PROMPT is not None else m["content"]) + SYSTEM_SUFFIX
        if m["role"] == "user" and DEPLOY_BUDGET and isinstance(m.get("content"), str):
            for k, v in DEPLOY_BUDGET.items():
                if k in BUDGET_RE:
                    m["content"] = re.sub(BUDGET_RE[k][0], BUDGET_RE[k][1].format(v), m["content"])
        if PAST_REASONING == "drop" and m["role"] == "assistant" and i != target_idx:
            m.pop("reasoning_content", None)
        out.append(m)
    return out


def render(tok, msgs, tools, gen, target_idx=None):
    return tok.apply_chat_template(clean(msgs, target_idx), tools=tools, tokenize=False, add_generation_prompt=gen,
                                   enable_thinking=True)


COMPACTION_TOL = 96   # tokens; set by --fidelity-tol
ERR_RE = re.compile(r'"status"\s*:\s*"error"')


def broken_safetensors(d):
    """Files whose safetensors header does not cover the file exactly. Kaggle's gemma-4-12b-it-qat-q4_0-unquantized v2 failed
    here on Oct 15 ("incomplete metadata, file not fully covered"); the Hugging Face copy is intact. [] = all good."""
    bad = []
    for f in sorted(Path(d).glob("*.safetensors")) if Path(d).is_dir() else []:
        try:
            with open(f, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                hdr = json.loads(fh.read(n))
            end = max(v["data_offsets"][1] for k, v in hdr.items() if k != "__metadata__")
            if f.stat().st_size != 8 + n + end:
                bad.append(f"{f.name}: file {f.stat().st_size} bytes, header covers {8 + n + end}")
        except Exception as e:
            bad.append(f"{f.name}: unreadable header ({type(e).__name__}: {e})")
    return bad


def pick_base(spec):
    """--base may list several candidates (comma-separated, e.g. every attached version); the first intact one is used."""
    for c in [x.strip() for x in spec.split(",") if x.strip()]:
        bad = broken_safetensors(c)
        if not bad:
            return c
        print(f"[base] skipping {c}: broken checkpoint: {bad[:3]}", flush=True)
    sys.exit(f"no intact base checkpoint among {spec!r}: attach another version or the Hugging Face copy")


def turn_errored(msgs, i):
    """True if assistant turn i's tool result is an error from a tool other than run_command (SWE-Lego error masking)."""
    nxt = msgs[i + 1] if i + 1 < len(msgs) else None
    return bool(nxt and nxt["role"] == "tool" and nxt.get("name") != "run_command"
                and ERR_RE.search(str(nxt.get("content", ""))))


def compacted_context(msgs, i, comps):
    """The conversation the model was sent at assistant turn i, as ADK 1.36 builds it (contents._process_compaction_events):
    every compaction made before the turn, minus those subsumed by a wider one, replaces the events in [start, end] with its
    summary as a model turn placed at `end`. The system prompt is not an event and always stays."""
    t = msgs[i].get("_ts")
    live = [c for c in comps if t is not None and c["created"] <= t]
    live = [c for c in live if not any(o is not c and o["start"] <= c["start"] and o["end"] >= c["end"]
                                       and (o["start"] < c["start"] or o["end"] > c["end"]) for o in live)]
    if not live:
        return msgs[:i]
    head = [m for m in msgs[:i] if m["role"] == "system"]
    rest = [(m["_ts"], k, m) for k, m in enumerate(msgs[:i]) if m["role"] != "system"
            and not any(c["start"] <= m["_ts"] <= c["end"] for c in live)]
    rest += [(c["end"], i + j, {"role": "assistant", "content": c["summary"]}) for j, c in enumerate(live)]
    return head + [m for _, _, m in sorted(rest, key=lambda x: (x[0], x[1]))]


def turn_samples(tok, traj, tools, stats):
    """[(prompt_text, target_text, turn_index, masked)] per assistant turn.

    The trace records each LLM call's real prompt size. Our rendering matches it to a constant offset; when the harness
    compacts the context (ADK EventsCompactionConfig: older events replaced by a summary), the prompt is rebuilt from the
    summaries run_arm.py logs (traj["compactions"], attached by build_rft_data.py). Without them (runs before Oct 15), or if
    the rebuilt prompt still drifts from the recorded size, the turn and every later one are not used."""
    msgs, out = traj["messages"], []
    comps = traj.get("compactions") or []
    stats["compactions_applied"] += len(comps)
    offset = None
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        ctx = compacted_context(msgs, i, comps) if comps else msgs[:i]
        prompt = render(tok, ctx, tools, True)
        rec = m.get("_prompt_tokens")
        if rec:
            d = len(tok(prompt, add_special_tokens=False)["input_ids"]) - rec
            offset = d if offset is None else offset
            if abs(d - offset) > COMPACTION_TOL:
                stats["after_compaction"] += sum(1 for x in msgs[i:] if x["role"] == "assistant")
                stats["compacted_trajectories"] += 1
                break
        full = render(tok, ctx + [m], tools, False, target_idx=len(ctx))
        if rec:                                  # fidelity: our prompt vs what the model was actually sent at this turn
            stats["prompt_tok_pairs"].append((len(tok(prompt, add_special_tokens=False)["input_ids"]), rec))
        if not full.startswith(prompt):
            if not str(m.get("reasoning_content") or "").strip():
                # thinking on: the generation prompt opens the thought channel, but a turn with no reasoning text renders
                # straight to its tool call or answer (converted external steps, e.g. read_file continuation chunks, final
                # answers written without a thought). It cannot be a
                # target without inventing a thought; it stays context only (no loss).
                stats["no_reasoning_context_only"] = stats.get("no_reasoning_context_only", 0) + 1
                continue
            stats["prefix_mismatch"] += 1
            if stats["prefix_mismatch"] <= 2:
                k = next((j for j, (x, y) in enumerate(zip(prompt, full)) if x != y), min(len(prompt), len(full)))
                stats["mismatch_examples"].append({"turn": i, "prompt_tail": prompt[max(0, k - 200):k + 200],
                                                   "full_at_same_place": full[max(0, k - 200):k + 200]})
            continue
        masked = ERROR_MASK and turn_errored(msgs, i)
        stats["masked_turns"] += masked
        out.append((prompt, full[len(prompt):], i, masked))
    return out


ERROR_MASK = True


def ids_and_positions(tok, text, spans):
    """Token ids of `text` and the positions of tokens that start inside one of the character `spans`."""
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    pos = [t for t, (a, b) in enumerate(enc["offset_mapping"]) if b > a and any(s <= a < e for s, e in spans)]
    return enc["input_ids"], pos


def window_samples(tok, traj, tools, stats, max_target, max_len):
    """Pack consecutive turns into windows (see the module docstring); returns [(ids, target_positions)]."""
    turns = turn_samples(tok, traj, tools, stats)
    out, cur = [], []

    def flush(group):
        if not group:
            return
        last_p, last_y = group[-1][0], group[-1][1]
        text = last_p + last_y
        spans, ok = [], True
        for p, y, _, masked in group:
            if not text.startswith(p + y):      # the window's text must contain every earlier turn verbatim
                ok = False
                break
            if not masked:
                spans.append((len(p), len(p) + len(y)))
        if ok and len(group) > 1:
            ids, pos = ids_and_positions(tok, text, spans)
            if pos and len(ids) <= max_len:
                out.append((ids, pos))
                stats["windows"] += 1
            elif len(ids) > max_len:
                stats["too_long"] += 1
            return
        if not ok:
            stats["window_fallback"] += 1
        for p, y, _, masked in group:           # one sample per turn
            if masked:
                continue
            ids, pos = ids_and_positions(tok, p + y, [(len(p), len(p) + len(y))])
            if pos and len(ids) <= max_len:
                out.append((ids, pos))
            elif len(ids) > max_len:
                stats["too_long"] += 1

    tgt = 0
    for t in turns:
        n = len(tok(t[1], add_special_tokens=False)["input_ids"])
        if n > max_target:                       # one turn alone over the budget: drop it (keeps it as context in later windows)
            stats["target_too_long"] += 1
            flush(cur); cur, tgt = [], 0
            continue
        if cur and (tgt + n > max_target or not t[0].startswith(cur[-1][0] + cur[-1][1])):   # budget, or a compaction
            flush(cur); cur, tgt = [], 0
        cur.append(t); tgt += n
    flush(cur)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--base", required=True, help="HF id or local dir of the unquantized QAT checkpoint; several, comma-separated, "
                    "= use the first whose safetensors are intact (broken_safetensors)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", action="store_true", help="render and count only (CPU)")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--grad-accum", type=int, default=4, help="windows per optimizer step (over all ranks)")
    ap.add_argument("--max-len", type=int, default=16384 + 2048, help="tokens per sample; longer samples are dropped")
    ap.add_argument("--max-target", type=int, default=2500, help="target tokens per window: the logits of a T-token target take "
                    "T x 262k x 2 bytes (bf16) plus a float32 chunk; 2,500 fits beside E4B on one L4, 31B split over 4 GPUs takes 8,000")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quant", choices=["4bit", "none"], default="4bit")
    ap.add_argument("--device-map", choices=["single", "auto"], default="single",
                    help="auto: one process, layers spread over every visible GPU (31B); single: one GPU per process")
    ap.add_argument("--max-memory-gib", type=float, default=19.5, help="per GPU, --device-map auto")
    ap.add_argument("--time-budget-h", type=float, default=0, help="0 = no limit")
    ap.add_argument("--no-eval-start", action="store_true", help="skip the val loss before training (saves one val pass)")
    ap.add_argument("--no-error-mask", action="store_true")
    ap.add_argument("--init-adapter", help="resume: continue training this LoRA (a previous session's adapter dir)")
    ap.add_argument("--stop-at", type=float, default=0, help="unix time: stop at a step boundary before it, save, exit 0")
    ap.add_argument("--stop-after-s", type=float, default=0, help="like --stop-at, counted from the first training step")
    ap.add_argument("--save-every-min", type=float, default=30, help="checkpoint the adapter (atomically) this often")
    ap.add_argument("--keep-order", action="store_true", help="train in the data file's order (its priority), shuffled only "
                    "within blocks of --order-block windows, so a run cut short has trained the file's head")
    ap.add_argument("--order-block", type=int, default=32)
    ap.add_argument("--tail-n", type=int, default=0, help="the file's last N trajectories are always trained, LAST: before the "
                    "deadline the run jumps to them in time (user: the model should see our competition data last)")
    ap.add_argument("--lr-floor", type=float, default=0.0, help="the LR decays to this fraction of the peak, not to 0 (so the "
                    "tail trained at the end still moves the model)")
    ap.add_argument("--system-prompt", help="file: replaces the recorded system prompt (train = deploy)")
    ap.add_argument("--deploy-budget", help='JSON, e.g. {"time_minutes": 5, "tool_calls": 30, "turns": 80, "timeout_seconds": 240}')
    ap.add_argument("--served-model", default="google/gemma-4-E4B-it-qat-w4a16-ct", help="written into adapter_config.json")
    ap.add_argument("--fidelity-tol", type=int, default=96, help="tokens: a turn whose rendered prompt departs from the recorded "
                    "size by more than this (beyond the first turn's offset) is after a compaction; it and later turns are dropped")
    ap.add_argument("--past-reasoning", choices=["keep", "drop"], default="keep")
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    global PAST_REASONING, COMPACTION_TOL, ERROR_MASK, SYSTEM_PROMPT, DEPLOY_BUDGET
    PAST_REASONING, COMPACTION_TOL, ERROR_MASK = a.past_reasoning, a.fidelity_tol, not a.no_error_mask
    SYSTEM_PROMPT = Path(a.system_prompt).read_text() if a.system_prompt else None
    DEPLOY_BUDGET = json.loads(a.deploy_budget) if a.deploy_budget else None
    from transformers import AutoTokenizer
    a.base = pick_base(a.base)
    print(f"[base] {a.base}", flush=True)
    tok = AutoTokenizer.from_pretrained(a.base)
    tools = harness_tool_schemas(B2_TOOLS)
    if RANK == 0:
        (out / "tools_rendered.json").write_text(json.dumps(tools, indent=1))
    trajs = [json.loads(l) for l in open(a.data) if l.strip()]
    tail_tasks = {t["instance_id"] for t in trajs[len(trajs) - a.tail_n:]} if a.tail_n else set()
    tasks = sorted({t["instance_id"] for t in trajs} - tail_tasks)   # the --tail-n trajectories are never held out
    random.Random(a.seed).shuffle(tasks)
    n_val = max(1, round(len(tasks) * a.val_frac)) if len(tasks) > 3 else 0
    val_tasks = set(tasks[:n_val])
    stats = {"prefix_mismatch": 0, "mismatch_examples": [], "too_long": 0, "prompt_tok_pairs": [], "after_compaction": 0,
             "compacted_trajectories": 0, "compactions_applied": 0, "masked_turns": 0, "windows": 0, "window_fallback": 0, "target_too_long": 0}
    splits = {"train": [], "val": []}
    owner = []                          # trajectory key of each train sample (resume bookkeeping)
    tkey = lambda t, k: t.get("source_traj_id") or f"{t['instance_id']}#{k}"
    for k, t in enumerate(trajs):
        for s in window_samples(tok, t, tools, stats, a.max_target, a.max_len):
            sp = "val" if t["instance_id"] in val_tasks else "train"
            splits[sp].append(s)
            if sp == "train":
                owner.append(tkey(t, k))
    rep = {"trajectories": len(trajs), "tasks": len(tasks), "val_tasks": sorted(val_tasks),
           "samples": {k: len(v) for k, v in splits.items()},
           "target_tokens": {k: sum(len(p) for _, p in v) for k, v in splits.items()},
           "sequence_tokens": {k: sum(len(i) for i, _ in v) for k, v in splits.items()},
           "max_sample_tokens": max((len(i) for v in splits.values() for i, _ in v), default=0),
           **{k: stats[k] for k in ("prefix_mismatch", "too_long", "target_too_long", "masked_turns", "windows", "window_fallback")},
           "no_reasoning_turns_context_only": stats.get("no_reasoning_context_only", 0),
           "base": a.base, "past_reasoning": a.past_reasoning, "system_prompt_replaced": bool(SYSTEM_PROMPT),
           "deploy_budget": DEPLOY_BUDGET,
           "turns_dropped_after_compaction": stats["after_compaction"], "trajectories_compacted": stats["compacted_trajectories"],
           "compaction_summaries_applied": stats["compactions_applied"]}
    pairs = stats["prompt_tok_pairs"]
    if pairs:   # rendered prompt tokens vs the trace's recorded prompt_tokens (same turn)
        d = sorted(r - o for o, r in pairs)
        rep["prompt_tokens_recorded_minus_rendered"] = {"n": len(d), "median": d[len(d) // 2], "p10": d[len(d) // 10],
                                                        "p90": d[9 * len(d) // 10]}
    if RANK == 0:
        (out / "check.json").write_text(json.dumps({**rep, "mismatch_examples": stats["mismatch_examples"]}, indent=1))
        print(json.dumps(rep, indent=1), flush=True)
    if stats["prefix_mismatch"]:
        sys.exit(f"STOP: {stats['prefix_mismatch']} turns do not continue their prompt under this chat template "
                 f"(see {out / 'check.json'}).")
    if a.check:
        print("check ok")
        return
    if not splits["train"]:
        sys.exit("STOP: no training samples")

    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(a.seed)
    ngpu = torch.cuda.device_count()
    gpu = LOCAL % ngpu   # more ranks than GPUs only in a smoke test (ranks share a GPU over gloo)
    torch.cuda.set_device(gpu)
    dev = f"cuda:{gpu}"
    if WORLD > 1:
        import torch.distributed as dist
        dist.init_process_group("nccl" if ngpu >= WORLD else "gloo")
    if a.device_map == "auto":
        if WORLD > 1:
            sys.exit("--device-map auto is one process; do not use torchrun")
        # "balanced", not "auto": "auto" fills GPU 0 to max_memory first, and the 31B weights (~19 GB in 4 bits) would all land
        # there, leaving no room for its activations. Balanced spreads the layers evenly; max_memory stays a hard cap.
        kw = dict(dtype=torch.bfloat16, device_map="balanced", attn_implementation="sdpa",
                  max_memory={i: f"{a.max_memory_gib}GiB" for i in range(ngpu)})
        dev = "cuda:0"
    else:
        kw = dict(dtype=torch.bfloat16, device_map={"": gpu}, attn_implementation="sdpa")
    if a.quant == "4bit":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    for r_ in range(WORLD):                  # one rank at a time: simultaneous loads could exhaust host RAM
        if r_ == RANK:
            model = AutoModelForCausalLM.from_pretrained(a.base, **kw)
        if WORLD > 1:
            dist.barrier()
    # NOT prepare_model_for_kbit_training: it casts every non-quantized weight to float32 (E4B's per-layer embeddings: +11 GB)
    for prm in model.parameters():
        prm.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    if RANK == 0:
        print(f"GPU memory after loading the base: {[round(torch.cuda.memory_allocated(i) / 1e9, 1) for i in range(ngpu)]} GB", flush=True)
    # every language-model projection, except k_proj on layers where K doubles as V (attention_k_eq_v, full attention: 31B)
    tc = model.config.get_text_config()
    n_layers, lt = tc.num_hidden_layers, getattr(tc, "layer_types", None) or []
    keq = [i for i in range(n_layers) if getattr(tc, "attention_k_eq_v", False) and i < len(lt) and lt[i] == "full_attention"]
    k_ok = "|".join(str(i) for i in range(n_layers) if i not in keq)
    target = (r".*language_model.*\.(q_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
              + (rf"|.*language_model.*\.layers\.({k_ok})\.self_attn\.k_proj" if k_ok else ""))
    if RANK == 0:
        print(f"k_eq_v layers without a k_proj LoRA: {keq}", flush=True)
    lcfg = LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=a.dropout, bias="none", task_type="CAUSAL_LM",
                      target_modules=target)
    if a.init_adapter:                  # resume: the previous session's LoRA, trainable
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.init_adapter, is_trainable=True, autocast_adapter_dtype=False)
        if RANK == 0:
            print(f"resumed from adapter {a.init_adapter}", flush=True)
    else:
        model = get_peft_model(model, lcfg, autocast_adapter_dtype=False)
    if RANK == 0:
        model.print_trainable_parameters()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    train = splits["train"]
    accum = max(1, round(a.grad_accum / WORLD))     # per rank; effective batch = accum x WORLD
    plan = {"total": math.ceil(int(len(train) * a.epochs) // WORLD / accum)}
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, plan["total"] // 10)) * max(a.lr_floor, 1 - s / max(1, plan["total"])))

    def loss_of(ids, pos):
        # logits only at the positions that predict a target token (logits_to_keep as an index tensor); CE in float32 chunks
        x = torch.tensor([ids], device=dev)
        # a plain list, not a tensor: accelerate's device_map hook moves tensor arguments to the FIRST GPU, while the output
        # layer sits on the last one (Kaggle nb19, Oct 15: "indices should be either on cpu or on the same device ... cuda:3");
        # the model indexes hidden_states[:, keep, :] with it unchanged
        keep = [p - 1 for p in pos]
        logits = model(input_ids=x, logits_to_keep=keep, use_cache=False).logits[0]
        y = torch.tensor([ids[p] for p in pos], device=logits.device)
        tot = 0.0
        for c in range(0, len(pos), 1024):
            tot = tot + F.cross_entropy(logits[c:c + 1024].float(), y[c:c + 1024], reduction="sum")
        return tot / len(pos)

    def evaluate():
        if not splits["val"]:
            return None
        model.eval()
        with torch.no_grad():
            v = [loss_of(i, p).item() for i, p in splits["val"][RANK::WORLD]]
        model.train()
        t = torch.tensor([sum(v), len(v)], device=dev, dtype=torch.float64)
        if WORLD > 1:
            dist.all_reduce(t)
        return (t[0] / max(t[1], 1)).item()

    log = open(out / "train_log.jsonl", "a") if RANK == 0 else open(os.devnull, "w")
    rng, step, t0 = random.Random(a.seed), 0, time.time()
    order = []
    while len(order) < len(train) * a.epochs:
        o = list(range(len(train)))
        if a.keep_order:                # the file's priority order, shuffled only inside small blocks
            o = [x for b in range(0, len(o), a.order_block) for x in rng.sample(o[b:b + a.order_block], len(o[b:b + a.order_block]))]
        else:
            rng.shuffle(o)
        order += o
    order = order[:int(len(train) * a.epochs)]
    tail_keys = set()
    if a.tail_n:                        # the last N trajectories of the file: always trained, at the very end
        keys_in_order = list(dict.fromkeys(owner))
        tail_keys = set(keys_in_order[-a.tail_n:])
        tail = [j for j in order if owner[j] in tail_keys]
        order = [j for j in order if owner[j] not in tail_keys] + tail
    order = order[RANK::WORLD][:len(order) // WORLD]     # same shuffled order on every rank; equal shard lengths
    tail_start = len(order) - sum(1 for j in order if owner[j] in tail_keys)
    model.train()
    if not a.no_eval_start:
        v0 = evaluate()
        if RANK == 0:
            print(f"val loss before training: {v0}", flush=True)
        log.write(json.dumps({"step": 0, "val_loss": v0}) + "\n")
    import signal
    stop = {"why": None}
    def _on_signal(sig, frm):
        stop["why"] = f"signal {sig}"
    for sg in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sg, _on_signal)
    done_i, seq_tok, last_save = 0, 0, time.time()
    if a.stop_after_s:
        a.stop_at = min(a.stop_at or float("inf"), time.time() + a.stop_after_s)

    def save(dest_name="adapter"):
        """Atomic: write to a temp dir, then rename over the old one, with the trajectories trained so far."""
        if RANK != 0:
            return
        tmp, dst = out / (dest_name + ".tmp"), out / dest_name
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
        model.save_pretrained(tmp)
        cp = tmp / "adapter_config.json"
        c_ = json.loads(cp.read_text()); c_["base_model_name_or_path"] = a.served_model; cp.write_text(json.dumps(c_, indent=2))
        seen = {}
        for j_ in order[:done_i]:
            seen[owner[j_]] = seen.get(owner[j_], 0) + 1
        need = {}
        for o_ in owner:
            need[o_] = need.get(o_, 0) + 1
        (tmp / "trained_ids.json").write_text(json.dumps(sorted(k for k, n in seen.items() if n >= need[k])))
        shutil.rmtree(dst, ignore_errors=True)
        tmp.rename(dst)

    run, i = 0.0, 0
    while i < len(order):
        if tail_keys and a.stop_at and i < tail_start and i % accum == 0 and i >= accum:
            per_sample = (time.time() - t0) / i
            if time.time() + 1.3 * per_sample * (len(order) - tail_start + accum) > a.stop_at:
                if RANK == 0:
                    print(f"deadline: skipping {tail_start - i} main samples to train the {len(order) - tail_start} tail samples",
                          flush=True)
                order = order[:i] + order[tail_start:]
                tail_start = i
        j = order[i]
        i += 1
        loss = loss_of(*train[j]) / accum
        loss.backward()
        run += loss.item()
        seq_tok += len(train[j][0])
        if i % accum == 0 or i == len(order):
            if WORLD > 1:                    # average LoRA gradients over ranks (equal shard lengths: every rank steps together)
                for p in params:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    dist.all_reduce(p.grad)
                    p.grad.div_(WORLD)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            done_i = i
            if a.time_budget_h and step == 2:   # fit the run into the budget; same decision on every rank
                per_step = (time.time() - t0) / step
                fit = torch.tensor([int(a.time_budget_h * 3600 * 0.9 / per_step)], device=dev)
                if WORLD > 1:
                    dist.broadcast(fit, 0)
                if fit.item() < plan["total"]:
                    plan["total"] = max(step + 1, fit.item())
                    if tail_keys:               # keep the tail: cut the main part instead
                        keep = max(i, plan["total"] * accum - (len(order) - tail_start))
                        order = order[:min(tail_start, keep)] + order[tail_start:]
                        tail_start = min(tail_start, keep)
                    else:
                        order = order[:plan["total"] * accum]
                    if RANK == 0:
                        print(f"time budget {a.time_budget_h} h at {per_step:.0f} s/step: {plan['total']} steps "
                              f"({plan['total'] * accum * WORLD / len(train):.2f} epochs)", flush=True)
            rec = {"step": step, "of": plan["total"], "train_loss": round(run, 4), "lr": sched.get_last_lr()[0],
                   "elapsed_s": round(time.time() - t0), "seq_tokens": seq_tok,
                   "tok_per_s": round(seq_tok / max(1e-9, time.time() - t0), 1),
                   "gpu_mem_gb": [round(torch.cuda.max_memory_allocated(g) / 1e9, 1) for g in range(ngpu)]}
            if i >= len(order):
                rec["val_loss"] = evaluate()
            if RANK == 0:
                print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()
            run = 0.0
            per = (time.time() - t0) / step
            if a.stop_at and time.time() + 1.5 * per > a.stop_at:      # the next step would end past the deadline
                stop["why"] = stop["why"] or "deadline"
            if stop["why"]:
                if RANK == 0:
                    print(f"stopping early ({stop['why']}) after step {step}: saving", flush=True)
                break
            if a.save_every_min and time.time() - last_save > a.save_every_min * 60:
                save(); last_save = time.time()
    if WORLD > 1:
        dist.barrier()
        if RANK != 0:
            dist.destroy_process_group()
            return
    ad = out / "adapter"
    save()                                               # base_model_name_or_path = the served model, + trained_ids.json
    (out / "train_meta.json").write_text(json.dumps({**rep, "args": vars(a), "steps": step, "k_eq_v_layers": keq,
                                                      "stopped": stop["why"], "seq_tokens": seq_tok,
                                                      "tok_per_s": round(seq_tok / max(1e-9, time.time() - t0), 1),
                                                      "seconds": round(time.time() - t0)}, indent=1))
    print(f"adapter saved: {ad}")
    if WORLD > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
