"""Build Kaggle notebooks 15-17 from notebook 14 (the new-build B2 runner), changing only what each arm needs.

    python scripts/planB/notebook_builder/make_nb15_17.py

15  B2-SUB  B2 on the union of the two subsets below (new build): the paired baseline for 16 and 17
16  B4      push: the structural brief prepended to the problem statement, on the 28 brief-eligible tasks
17  O8K     max_output_tokens 8192, on the 21 context-heavy tasks

Runner changes (every one asserted to apply exactly once, so a changed template fails loudly):
- CONFIG: TASK_SCOPE "list" + TASK_LIST; BRIEFS (B4 only); VARIANT_TAG / MAX_OUTPUT_TOKENS per arm;
- run_arm.py gains --briefs (scripts/planB/run_arm.py has the same code);
- validity re-check, task order, run loop and scorecard read TASK_LIST / BRIEFS.
build_run.py / gen_all.py are stale (HANDOFF section 7), so this edits the last notebook that ran on the new build.
"""
import copy
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NB = ROOT / "planB_results/kaggle_upload/notebooks"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from patch_run_arm import patch as patch_run_arm  # noqa: E402

DEV = (ROOT / "scripts/planB/official_slices/dev_slice.txt").read_text().split()
HOLDOUT = set((ROOT / "scripts/planB/official_slices/holdout_slice.txt").read_text().split())
BRIEFS = json.loads((ROOT / "planB_results/kaggle_upload/b4_briefs/briefs.json").read_text())
order = lambda ids: sorted(ids, key=lambda i: (i not in DEV, i))  # noqa: E731  dev tasks first, then the rest
# valid on Kaggle (notebook 01), not holdout, ended context_exceeded in B2_s1 (nb01), B2-WH2 (nb14) or FC (nb03)
CTX = order(["fastapi_12942", "fastapi_13786", "fastapi_14246", "fastapi_14419", "fastapi_14485", "fastapi_14962",
             "fastapi_15280", "fastapi_15661", "fastapi_15800", "fastapi_5077", "fastapi_9555", "rich_3043", "rich_3105",
             "rich_3468", "rich_3521", "rich_3782", "rich_3934", "rich_3942", "rich_3944", "rich_3953", "rich_4075"])
B4 = order(BRIEFS)   # valid on Kaggle, not holdout, non-empty brief (build_b4_briefs.py)
UNION = order(set(CTX) | set(B4))
assert not HOLDOUT & set(UNION)

INPUTS = ("competition data, `metric/gemma-4-developer-agent-wheelhouse` (current version), model `google/gemma-4` "
          "`gemma-4-31b-it-qat-w4a16-ct` v2, **notebook 01's output**, the **`graphcomp-planb`** dataset, and for a resumed "
          "run **this notebook's own previous version output**. GPU L4 × 4. Save Version → Save & Run All.")
TAIL = "\n\nEverything below is the shared Plan B runner; only the CONFIG values differ between notebooks.\n\n---"
SPECS = [
    dict(file="15_B2_subset_newbuild_s1.ipynb", tag="B2-SUB", out=16384, tasks=UNION, briefs={},
         md=f"""# 15 — B2 on the B4 + context-heavy subset (new build), seed 1

**What this run is:** the B2 condition exactly (as notebook 14) on {len(UNION)} tasks: the {len(B4)} tasks that get a B4 brief and the {len(CTX)} context-heavy tasks ({len(set(B4) & set(CTX))} in both). Every task is valid on Kaggle (notebook 01) and none is in the holdout slice.

**Compared with:** 16 (B4) and 17 (O8K), task by task: it is their paired baseline on the new build. Its {sum(t in DEV for t in UNION)} dev tasks also repeat notebook 14, a second B2 sample for the noise bar.

**Tasks:** {len(UNION)}, dev-slice tasks first (about 2.5 h).

**Inputs:** {INPUTS}{TAIL}"""),
    dict(file="16_B4_push_s1.ipynb", tag="B4", out=16384, tasks=B4, briefs={k: BRIEFS[k] for k in B4},
         md=f"""# 16 — B4 push: structural brief in the problem statement, seed 1

**What this run is:** B2 plus the pre-registered B4 brief (Plan B table: 1-hop callers and callees of the code elements the issue names, tests filtered, grouped by file), prepended to the problem statement. Briefs are built on CPU by `scripts/planB/build_b4_briefs.py` from the issue text and the provided graph only, and are embedded below; tasks whose issue names no resolvable element get none, so the run uses only the {len(B4)} tasks that get one (23 of whose briefs name a gold file).

**Compared with:** 15 (B2 on the same tasks, same build), task by task. Primary: steps to the first gold-file read and resolution (McNemar). On the dev slice B2 already resolves all 8 valid briefed tasks, so this run uses non-holdout tasks outside the dev slice too.

**Tasks:** {len(B4)} (about 1.5 h).

**Inputs:** {INPUTS}{TAIL}"""),
    dict(file="17_O8K_ctxheavy_s1.ipynb", tag="O8K", out=8192, tasks=CTX, briefs={},
         md=f"""# 17 — output reserve 8192 on the context-heavy tasks, seed 1

**What this run is:** B2 with `max_output_tokens` 8192 instead of 16384, so the agent gets about 24k tokens of input instead of 16k (the pre-registered O8K arm). Run on the {len(CTX)} tasks that hit the context limit in a 31B run (B2_s1, B2-WH2 or FC), because on the dev slice it can move at most 2 tasks.

**Compared with:** 15 (B2 on the same tasks, same build), task by task. Expected: fewer context-exceeded endings; watch thinking truncation and waste.

**Tasks:** {len(CTX)} (about 1.5 h).

**Inputs:** {INPUTS}{TAIL}"""),
]


def rep(src, old, new):
    assert src.count(old) == 1, f"template changed, not found exactly once: {old[:80]!r}"
    return src.replace(old, new)


def build(spec, tmpl):
    nb = copy.deepcopy(tmpl)
    cells = [''.join(c["source"]) for c in nb["cells"]]
    cells[0] = spec["md"]
    c = cells[2]
    c = re.sub(r'VARIANT_TAG = "[^"]*"', f'VARIANT_TAG = "{spec["tag"]}"', c, count=1)
    c = re.sub(r"MAX_OUTPUT_TOKENS = \d+", f"MAX_OUTPUT_TOKENS = {spec['out']}", c, count=1)
    c = rep(c, 'TASK_SCOPE = "dev"            # "all" = dev slice, then non-slice tasks, then holdout; "dev" = dev slice only\n',
            'TASK_SCOPE = "list"           # "all" = dev slice, then non-slice tasks, then holdout; "dev" = dev slice only; "list" = TASK_LIST\n'
            f"TASK_LIST = {spec['tasks']!r}\n")
    c = rep(c, "PROMPT_ADDENDUM = ''\n", "PROMPT_ADDENDUM = ''\n"
            "# B4 only: {instance_id: {brief, problem_sha256}} from scripts/planB/build_b4_briefs.py, prepended to the problem statement\n"
            f"BRIEFS = {json.dumps(spec['briefs'])}\n")
    cells[2] = c
    cells[9] = rep(cells[9], '"TASK_SCOPE": TASK_SCOPE, ', '"TASK_SCOPE": TASK_SCOPE, "TASK_LIST": TASK_LIST, "N_BRIEFS": len(BRIEFS), ')
    head, body = cells[11].split("\n", 1)
    cells[11] = head + "\n" + patch_run_arm(body)
    cells[16] = cells[16] + (
        "\n# B4: write the briefs for run_arm.py --briefs and scorecard.py --brief-dir; stop before serving if a problem statement\n"
        "# here differs from the one the brief was built from (the brief would then describe another issue text)\n"
        "if BRIEFS:\n"
        "    import hashlib\n"
        "    _bad = [i for i, b in BRIEFS.items() if hashlib.sha256(tasks_by_id[i]['problem_statement'].encode()).hexdigest() != b['problem_sha256']]\n"
        "    if _bad:\n"
        "        raise SystemExit(f'STOP: problem statements differ from the briefs\\' source for {_bad}. Nothing has run on the GPU yet.')\n"
        "    (GC / 'briefs.json').write_text(json.dumps(BRIEFS))\n"
        "    (OUT / 'provenance/briefs').mkdir(parents=True, exist_ok=True)\n"
        "    for _i, _b in BRIEFS.items():\n"
        "        (OUT / 'provenance/briefs' / f'{_i}.md').write_text(_b['brief'])\n"
        "    plog(f'B4 briefs: {len(BRIEFS)} tasks, problem statements match; written to {GC / \"briefs.json\"}')\n")
    cells[17] = rep(cells[17], '(OFFICIAL_DEV if TASK_SCOPE == "dev" else OFFICIAL_DEV + OFFICIAL_HOLDOUT)',
                    '(TASK_LIST if TASK_SCOPE == "list" else OFFICIAL_DEV if TASK_SCOPE == "dev" else OFFICIAL_DEV + OFFICIAL_HOLDOUT)')
    cells[18] = rep(cells[18], '    ORDER = DEV + (rest + HOLDOUT if TASK_SCOPE == "all" else [])\n',
                    '    ORDER = DEV + (rest + HOLDOUT if TASK_SCOPE == "all" else [])\n'
                    '    if TASK_SCOPE == "list":\n'
                    '        assert not set(TASK_LIST) & set(HOLDOUT), "TASK_LIST must not touch the holdout slice"\n'
                    '        ORDER = list(TASK_LIST)\n')
    cells[24] = rep(cells[24], '--model-alias {MODEL_SLUG}",\n',
                    '--model-alias {MODEL_SLUG}"\n'
                    '               + (f" --briefs {GC / \'briefs.json\'}" if BRIEFS else ""),\n')
    cells[28] = rep(cells[28], 'f"--out {OUT / \'scorecard\' / RUN_NAME} --arm {ARM}"',
                    'f"--out {OUT / \'scorecard\' / RUN_NAME} --arm {ARM}"\n'
                    '           + (f" --brief-dir {OUT / \'provenance/briefs\'}" if BRIEFS else "")')
    for cell, src in zip(nb["cells"], cells):
        cell["source"] = src.splitlines(keepends=True)
        if cell["cell_type"] == "code":
            cell["outputs"], cell["execution_count"] = [], None
    return nb


if __name__ == "__main__":
    tmpl = json.loads((NB / "14_B2_newstack_dev_s1.ipynb").read_text())
    for s in SPECS:
        (NB / s["file"]).write_text(json.dumps(build(s, tmpl), indent=1, ensure_ascii=False) + "\n")
        print(f"{s['file']}: {s['tag']}, {len(s['tasks'])} tasks, {len(s['briefs'])} briefs, max_output_tokens {s['out']}")
