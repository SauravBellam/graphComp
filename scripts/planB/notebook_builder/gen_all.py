"""Generate one Kaggle notebook per Plan B job (Oct 12), all from build_run.py."""
import json
import os
import subprocess
import sys
from pathlib import Path

FORCED = """

## Graph tools (required in this run)
In every turn, before any other tool call, call one of `get_code_neighbors`, `search_similar_code` or `get_code_subgraph`.
Do this on every turn, even when you already know which file to read or edit.
"""
QUALIFIED = """

## Naming code in graph tools
When you call `get_code_neighbors` or `get_code_subgraph`, name each function, method or class by its fully qualified name in
the form `package.module.Class.method` (for example `rich.table.Table.add_row`), not by a bare name such as `add_row`. If you
do not know the qualified name yet, find it first (for example with `search_similar_code`, or by reading the file).
"""

ATTACH_01 = ("**Inputs:** competition data, `metric/gemma-4-developer-agent-wheelhouse`, model `google/gemma-4` "
             "`gemma-4-31b-it-qat-w4a16-ct` v2, **notebook 01's output** (validity, slices, package list), and for a resumed run "
             "**this notebook's own previous version output**. GPU L4 × 4. Save Version → Save & Run All.")

def job(n, title, what, pairs, scope_note, attach=ATTACH_01):
    return (f"# {n} — {title}\n\n**What this run is:** {what}\n\n**Compared with:** {pairs}\n\n**Tasks:** {scope_note}\n\n{attach}\n\n"
            "Everything below is the shared Plan B runner; only the CONFIG values differ between notebooks.\n\n---")

ALL = "all 129, dev slice first, then the non-slice tasks, then holdout. Re-run with the previous version attached until the summary says ARM COMPLETE."
DEV = "the 24 dev-slice tasks only (about 1.5 to 2 h)."

VARIANTS = [
    dict(file="01_validity_B2_s1.ipynb", tag="B2", arm="B2", seed=1, scope="all", validity=True,
         title="graphcomp 01 validity B2 s1",
         job=job("01", "validity + B2 baseline, seed 1",
                 "first the validity check in this environment (about 40 min: one gold-patch pass over the 115 fastapi/rich tasks, "
                 "then an empty-patch pass over the ones the gold patch resolves), then the B2 baseline (all nine tools, sample prompt).",
                 "every other notebook: they all read validity and the slices from this one's output.", ALL,
                 attach="**Inputs:** competition data, `metric/gemma-4-developer-agent-wheelhouse`, model `google/gemma-4` "
                        "`gemma-4-31b-it-qat-w4a16-ct` v2; for a resumed run, this notebook's previous version output. Do not attach "
                        "the old CPU B1 output. GPU L4 × 4. Save Version → Save & Run All. **Run this one first.**")),
    dict(file="02_B3_s1.ipynb", tag="B3", arm="B3", seed=1, scope="all", validity=False,
         title="graphcomp 02 B3 s1",
         job=job("02", "B3 no graph tools, seed 1", "the three graph tools removed from the agent; everything else as B2.",
                 "01 (B2 s1), task by task: the pull benefit and the counterfactual need label v2.", ALL)),
    dict(file="03_forced_call_dev_s1.ipynb", tag="FC", arm="B2", seed=1, scope="dev", validity=False, addendum=FORCED,
         title="graphcomp 03 forced call dev s1",
         job=job("03", "forced graph call (anti-Goodhart arm), dev slice, seed 1",
                 "B2 with one paragraph added to the prompt that demands a graph call every turn (Plan B 4.4 test 4).",
                 "01's dev-slice tasks (B2 s1). A scorecard version must score this run lower than B2.", DEV)),
    dict(file="04_qualified_names_dev_s1.ipynb", tag="QN", arm="B2", seed=1, scope="dev", validity=False, addendum=QUALIFIED,
         title="graphcomp 04 qualified names dev s1",
         job=job("04", "qualified names in graph-tool arguments, dev slice, seed 1",
                 "B2 with one paragraph asking for `package.module.Class.method` arguments (the Plan A misresolution fix; C2a's first variant).",
                 "01's dev-slice tasks: resolution precision should rise toward 95%.", DEV)),
    dict(file="05_B2_s2.ipynb", tag="B2", arm="B2", seed=2, scope="all", validity=False, title="graphcomp 05 B2 s2",
         job=job("05", "B2 baseline, seed 2", "B2 exactly as 01, seed 2.", "06 (B3 s2); with 01 and 07 for seed variance and pass^3.", ALL)),
    dict(file="06_B3_s2.ipynb", tag="B3", arm="B3", seed=2, scope="all", validity=False, title="graphcomp 06 B3 s2",
         job=job("06", "B3 no graph tools, seed 2", "B3 exactly as 02, seed 2.", "05 (B2 s2).", ALL)),
    dict(file="07_B2_s3.ipynb", tag="B2", arm="B2", seed=3, scope="all", validity=False, title="graphcomp 07 B2 s3",
         job=job("07", "B2 baseline, seed 3", "B2 exactly as 01, seed 3.", "08 (B3 s3).", ALL)),
    dict(file="08_B3_s3.ipynb", tag="B3", arm="B3", seed=3, scope="all", validity=False, title="graphcomp 08 B3 s3",
         job=job("08", "B3 no graph tools, seed 3", "B3 exactly as 02, seed 3.", "07 (B2 s3).", ALL)),
    dict(file="09_thinking0_dev_s1.ipynb", tag="T0", arm="B2", seed=1, scope="dev", validity=False, think=0,
         title="graphcomp 09 thinking0 dev s1",
         job=job("09", "thinking budget 0, dev slice, seed 1", "B2 with `thinking_budget` 0 instead of 4096 (Plan B section 6).",
                 "01's dev-slice tasks. Expected: necessity match moves, resolution precision does not.", DEV)),
    dict(file="10_outreserve8k_dev_s1.ipynb", tag="O8K", arm="B2", seed=1, scope="dev", validity=False, max_out=8192,
         title="graphcomp 10 outreserve8k dev s1",
         job=job("10", "output reserve 8192, dev slice, seed 1",
                 "B2 with `max_output_tokens` 8192 instead of 16384, so the agent gets about 24k of input instead of 16k "
                 "(Plan B section 6 context ablation; B0 found tasks dying at the 16k input wall).",
                 "01's dev-slice tasks. Expected: fewer context-exceeded endings; waste should rise before localization falls.", DEV)),
]

# Size sweep (Oct 13: smaller QAT w4a16 variants allowed). B2 condition exactly, model changed; dev slice; sample LoRAs
# (31B adapters) not mounted. Notebook 11 (12B, architecture gemma4_unified) needs a serving check.
def attach_model(slug):
    return (f"**Inputs:** competition data, `metric/gemma-4-developer-agent-wheelhouse`, model `google/gemma-4` `{slug}` (only this "
            "model), **notebook 01's output**, the **`graphcomp-planb`** dataset, and for a resumed run this notebook's own previous "
            "version output. GPU L4 × 4. Save Version → Save & Run All.")
for _n, _m, _slug in (("11", "12b", "gemma-4-12b-it-qat-w4a16-ct"), ("12", "e4b", "gemma-4-e4b-it-qat-w4a16-ct"),
                      ("13", "e2b", "gemma-4-e2b-it-qat-w4a16-ct")):
    VARIANTS.append(dict(file=f"{_n}_B2_{_m}_dev_s1.ipynb", tag=f"B2-{_m.upper()}", arm="B2", seed=1, scope="dev", validity=False,
                         model=_m, title=f"graphcomp {_n} B2 {_m} dev s1",
                         job=job(_n, f"B2 on {_slug} (size sweep), dev slice, seed 1",
                                 f"the B2 condition exactly, with `{_slug}` instead of 31B; sample LoRAs not mounted (31B adapters).",
                                 "01's dev-slice tasks (B2 s1, 31B), task by task; the scorecard's cross-model ordering check.",
                                 "the 24 dev-slice tasks only.", attach=attach_model(_slug))))

# Oct 13: the wheelhouse was re-uploaded (same versions, rebuilt files; old versions not selectable). Notebook 14 is B2 on the
# new build, the baseline for every arm run on it (the runner tags each run with its stack build; see provenance/stack.json).
VARIANTS.append(dict(file="14_B2_newstack_dev_s1.ipynb", tag="B2-WH2", arm="B2", seed=1, scope="dev", validity=False,
                     title="graphcomp 14 B2 newstack dev s1",
                     job=job("14", "B2 on the rebuilt wheelhouse (Oct 13), dev slice, seed 1",
                             "the B2 condition exactly, on the wheelhouse as re-uploaded on Oct 13; the baseline for every arm on that build.",
                             "B2_s1's dev-slice tasks (old build): the build effect alone. Every other arm on the new build pairs with this run.",
                             "the 24 dev-slice tasks only; validity re-checked on them first (about 10 min, CPU).",
                             attach=ATTACH_01 + " Also attach the `graphcomp-planb` dataset.")))

if __name__ == "__main__":
    here = Path(__file__).parent
    for v in VARIANTS:
        env = {**os.environ, "NB_VARIANT": json.dumps(v)}
        r = subprocess.run([sys.executable, str(here / "build_run.py")], env=env, capture_output=True, text=True)
        print(r.stdout.strip() or r.stderr[-2000:])
    (here / "variants.json").write_text(json.dumps(VARIANTS, indent=1))
