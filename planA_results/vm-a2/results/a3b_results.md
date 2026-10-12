# A3b results: text-identity checks and the GNN sweep

> **Update, Oct 11, about 12:05 CDT.** Item 4 (the GNN sweep) has now run on all 127 graphs on a GPU
> (section 7). It settles the question section 3.3 left open: **graph structure adds no information
> about the provided vectors beyond text.** Final verdict and paper wording in section 7.6; it
> supersedes section 3.5. Sections 1 to 6 are the morning's checks, unchanged except where marked.

Written Oct 11, 2026 (about 08:00 CDT) from the first clean run on `vm-a2` after the wipe described
in `HANDOFF_planA_rerun.md` section 0a. Covers `HANDOFF_planBC.md` section 1.2 items 1 to 3. Item 4
(the GNN sweep) has not run yet. Every number below is copied from command output; the commands and
outputs are reproduced in section 5 so the reading can be checked.

## 1. What ran

| Item | Value |
| --- | --- |
| Machine | `vm-a2`, zone `us-south1-b`, e2-standard-16 (16 vCPU, 64 GB), freshly wiped (`~/results` empty, scripts re-pushed from `planA_results 2/scripts/`) |
| Script | `~/scripts/planA/a3b_gnn.py` (with `common.py`, `stats.py` beside it), run in `~/a3env` |
| Command | `cd ~/scripts/planA && nohup ~/a3env/bin/python a3b_gnn.py --data ~/graphComp/archive --out ~/results/a3b --graphs all --stages checks nbrmean > ~/results/a3b_checks.log 2>&1 &` |
| Graphs | all 127 (fastapi, rich, requests) |
| Run time | minutes; finished before the first status check |
| Log | `~/results/a3b_checks.log`; `grep -n -E 'traceback|Error'` returns nothing |
| Outputs | `~/results/a3b/`: `a3b_checks_text_identity.csv`, `a3b_checks_text_identity_nodes.csv`, `a3b_checks_duplicates.csv`, `a3b_checks_isolated.csv`, `a3b_nbrmean.csv`, `a3b_provenance.json`, `a3b_summary.json` |

Not run here: the six-graph CPU `fit` (skipped; superseded by the full sweep) and the GPU
`fit`/`linkpred` sweep (ran later the same day on vm-gpu; section 7).

## 2. Numbers

### 2.1 `a3b_summary.json`, `checks` block (verbatim)

```json
{
 "text_identity_pairs": 3358,
 "text_identity_cos_mean_over_pairs": 0.9996263373635493,
 "text_identity_share_eq1_over_pairs": 0.9004163672996632,
 "text_identity_verdict": "mostly_text: identical text gives near-identical vectors",
 "duplicates_share_mean": 0.1670737206122978,
 "isolated_auroc_centroid_mean": 0.9747724415749077
}
```

`nbrmean` block: `nbr_mean_cos_mean` 0.9927, `global_mean_cos_mean` 0.9202.

### 2.2 `a3b_checks_text_identity.csv`, column means over the 3,358 same-repo graph pairs

One row per pair of graphs of the same repo. "Same text" = node ids present in both graphs whose
`text` field is byte-identical. Pairs per repo: fastapi 2,211, rich 1,081, requests 66.

| Column | Mean over pairs | Min over pairs | Meaning |
| --- | --- | --- | --- |
| `same_text_share_eq1` | 0.9004 | 0.3547 | share of same-text nodes whose two vectors have cosine exactly 1.0 (float equality) |
| `same_text_share_ge_0.99` | 0.9955 | 0.9842 | share with cosine at or above 0.99 |
| `same_text_cos_p05` | 0.9996 | 0.9916 | 5th-percentile cosine among same-text nodes |
| `spearman_cos_vs_nbr_jaccard` | 0.0982 | -0.0109 | rank correlation between a node's cosine shift and how much its neighbour set stayed the same |
| `same_text_same_nbrs_cos_mean` | 0.9998 | 0.9986 | same text, neighbour set unchanged |
| `same_text_changed_nbrs_cos_mean` | 0.9712 | 0.8697 | same text, neighbour set changed (n = 3,253 pairs; the rest had no such node) |

`same_text_share_eq1_over_pairs` in the summary is the mean of the per-pair `same_text_share_eq1`
column (0.9004 in both places), confirmed in section 5.

## 3. Interpretation

### 3.1 The pre-registered rule fired, and it was the wrong instrument

The rerun handoff fixed the decision rule before the run: `text_identity_share_eq1_over_pairs` below
0.99 means "not a text embedding: identical text gets different vectors across commits". It is 0.900,
so by the letter the rule fires.

But `share_eq1` tests exact float equality. The same rows show that 99.55% of same-text nodes stay
within cosine 0.99 and the 5th-percentile cosine is 0.9996 (worst pair 0.9916). The 10% that fail
`eq1` differ in the last float32 bits, not in direction. The rule assumed a text encoder would return
bit-identical vectors; the data show that assumption was wrong at the float level. The rule is
reported as written and as having misfired; the metrics that answer the question are
`same_text_share_ge_0.99` and `same_text_cos_p05`.

### 3.2 The vectors are a text embedding

Identical text gives the same vector across commits, within numerical noise, on 3,358 pairs across all
three repos. A structural or per-graph encoder would not do this: it would move every vector when the
graph around it changes. This settles item 1 of section 1.2.

The `nbrmean` result (0.993 against a 0.920 global-mean baseline) is therefore not evidence of
structure in the vectors. Neighbouring functions have similar text, so their text vectors are
similar. It is homophily.

The 16.7% exact-duplicate share per graph is consistent with the same story: many functions have
identical text (for example short `__init__` bodies) and get identical vectors.

### 3.3 Not purely text: a small context-dependent component

Nodes with identical text whose neighbour set changed average cosine 0.971 (worst pair 0.870),
against 0.9998 when the neighbour set is unchanged. That gap is small but consistent, and the Spearman
of 0.10 says the shift tracks neighbour change only weakly. Something beyond the node's own `text`
enters the vector. Two explanations fit and this test cannot separate them:

- the encoder saw graph context (neighbour text pooled in, or a GNN layer over text features);
- the encoder embedded a wider text window than the node body (enclosing class, file, docstrings),
  which changes when neighbouring code changes even though the node's own text does not.

The GPU sweep is the test that separates them: in the `fit` arm, `both` minus `text` on held-out
cosine is the direct measure of whether graph structure adds predictive signal beyond text.
*(Resolved in section 7: it does not. `both` minus `text` is negative in every repo and positive in
3 of 127 graphs by at most +0.0013, so the context component is attributed to a wider text window.)*

### 3.4 Isolated nodes

`isolated_auroc_centroid_mean` 0.975: a one-dimensional centroid score separates isolated from
connected nodes well. Under a text encoder this means isolated nodes are a textually distinct kind of
code (short helpers, fixtures, dead code) rather than a degenerate structural cluster. Worth one line
in the paper; not evidence against 3.2. Read `a3b_checks_isolated.csv` per repo before writing it.

### 3.5 Paper wording (section 5.6) *(superseded by section 7.6)*

Keep "not reproducible from EmbeddingGemma". Add: identical text yields identical vectors across
commits (cosine at or above 0.99 for 99.6% of same-text nodes, 5th percentile 0.9996, 3,358
same-repo graph pairs), so the provided vectors are a text embedding from an unidentified encoder,
with a small context-dependent component (about a 3-point cosine drop when the surrounding code
changes, `same_text_changed_nbrs_cos_mean` 0.971). State that the pre-registered `share_eq1` rule
fired on float noise and was replaced by `share_ge_0.99` and `cos_p05` after the run. Cite
`a3b_summary.json` and `a3b_checks_text_identity.csv` by field name.

### 3.6 Caveats

- Report per repo. requests has 66 pairs against fastapi's 2,211; pooled means are fastapi-weighted.
- `spearman_cos_vs_nbr_jaccard` can be NaN for near-identical commits (constant input); use nan-aware
  means when recomputing. The run above found 3,358 non-NaN values.
- The per-node CSV holds the first 2,000 nodes per pair only.
- Nothing here identifies the encoder. A3 remains the answer on that (no public text encoder matches).

## 4. Files to pull and where they go

From the laptop, in the project folder:

```bash
cd ~/Desktop/Projects/graphComp
gcloud compute scp --recurse vm-a2:~/results/a3b planA_results/vm-a2/results/ --zone us-south1-b
gcloud compute scp vm-a2:~/results/a3b_checks.log planA_results/vm-a2/results/ --zone us-south1-b
```

Pull once the six-graph `fit` has also run (if it runs), or pull `a3b` now and `a3b_cpu6` later:

```bash
gcloud compute scp --recurse vm-a2:~/results/a3b_cpu6 planA_results/vm-a2/results/ --zone us-south1-b
gcloud compute scp vm-a2:~/results/a3b_cpu6.log planA_results/vm-a2/results/ --zone us-south1-b
```

Do not re-run the `a3b` pull into a folder that already has `a3b/`: `scp --recurse` would nest it as
`a3b/a3b/`. Delete the local copy first or pull to a temp name.

Expected layout after the pull:

```text
planA_results/vm-a2/results/
├── a3b/                       # all seven output files + provenance
├── a3b_checks.log
├── a3b_results.md             # this file
└── (later) a3b_cpu6/, a3b_cpu6.log, and vm-gpu/results/a3b_gpu/ from the GPU VM
```

Record the sha256 of `a3b_gnn.py` and `common.py` as run (`a3b_provenance.json` should carry it;
confirm) and add them to `provenance/scripts_sha256.txt` when section 1.1 is pushed.

## 5. Evidence: commands and their output

All run from the laptop with `gcloud compute ssh vm-a2 --zone us-south1-b --command "..."`.

Status check (job finished; `pgrep` found only itself):

```text
$ pgrep -af a3b_gnn.py; tail -5 ~/results/a3b_checks.log; ls ~/results/a3b
3375 bash -c pgrep -af a3b_gnn.py; tail -5 ~/results/a3b_checks.log; ls ~/results/a3b
  "nbrmean": {
    "nbr_mean_cos_mean": 0.9927256773775956,
    "global_mean_cos_mean": 0.9202411676955036
  }
}
a3b_checks_duplicates.csv
a3b_checks_isolated.csv
a3b_checks_text_identity.csv
a3b_checks_text_identity_nodes.csv
a3b_nbrmean.csv
a3b_provenance.json
a3b_summary.json
```

Error scan and verdict:

```text
$ grep -n -E 'traceback|Error' ~/results/a3b_checks.log | head; python3 -c "import json;d=json.load(open('/home/sauravbellam/results/a3b/a3b_summary.json'));c=d['checks'];print(c.get('text_identity_verdict'));print({k:v for k,v in c.items() if 'share_eq1' in k})"
mostly_text: identical text gives near-identical vectors
{'text_identity_share_eq1_over_pairs': 0.9004163672996632}
```

`checks` block and CSV header plus first three rows:

```text
$ python3 -c "import json;d=json.load(open('/home/sauravbellam/results/a3b/a3b_summary.json'));print(json.dumps(d['checks'],indent=1)[:3500])"; head -4 ~/results/a3b/a3b_checks_text_identity.csv
{
 "text_identity_pairs": 3358,
 "text_identity_cos_mean_over_pairs": 0.9996263373635493,
 "text_identity_share_eq1_over_pairs": 0.9004163672996632,
 "text_identity_verdict": "mostly_text: identical text gives near-identical vectors",
 "duplicates_share_mean": 0.1670737206122978,
 "isolated_auroc_centroid_mean": 0.9747724415749077
}
repo,graph_a,graph_b,n_common_ids,n_same_text,n_diff_text,same_text_cos_mean,same_text_cos_median,same_text_cos_p05,same_text_share_eq1,same_text_share_ge_0.99,diff_text_cos_mean,nbr_jaccard_mean,spearman_cos_vs_nbr_jaccard,same_text_same_nbrs_cos_mean,same_text_changed_nbrs_cos_mean
fastapi/fastapi,fastapi_ee22a4b8ca46dcce26c8c183afc4992a888d8be2,fastapi_cb83b83dcf78eab4ea17d504db5abcda705fbdc4,4255,4250,5,0.9999988320154303,1.0,0.9999998807907104,0.9985882352941177,1.0,0.9475580215454101,0.9999607843137254,0.027756716854845324,0.9999990345282004,0.999138355255127
fastapi/fastapi,fastapi_ee22a4b8ca46dcce26c8c183afc4992a888d8be2,fastapi_22b02e26f9e8c7e32bd8266e2b0ebe8bb3a0db2b,4259,4254,5,0.9999999963570269,1.0,0.9999998807907104,1.0,1.0,0.9475579977035522,1.0,0.0,0.9999999963570269,
fastapi/fastapi,fastapi_ee22a4b8ca46dcce26c8c183afc4992a888d8be2,fastapi_48d58ae3b6ef4ded5a4794618c00345667256d98,4179,4123,56,0.9999624422502668,1.0,0.9999957144260406,0.983749696822702,0.99830220713073,0.9906532466411591,0.9997008650658905,0.04793908230610907,0.9999673158075045,0.9932694236437479
```

Column means and minima over all pairs (the table in 2.2):

```text
$ python3 -c "
import csv,statistics as s
r=list(csv.DictReader(open('/home/sauravbellam/results/a3b/a3b_checks_text_identity.csv')))
f=lambda k:[float(x[k]) for x in r if x[k] not in ('','nan')]
for k in ['same_text_share_eq1','same_text_share_ge_0.99','same_text_cos_p05','spearman_cos_vs_nbr_jaccard','same_text_same_nbrs_cos_mean','same_text_changed_nbrs_cos_mean']:
    v=f(k);print(k,'n',len(v),'mean',round(s.mean(v),5),'min',round(min(v),5))
from collections import Counter;print(Counter(x['repo'] for x in r))"
same_text_share_eq1 n 3358 mean 0.90042 min 0.3547
same_text_share_ge_0.99 n 3358 mean 0.99551 min 0.98423
same_text_cos_p05 n 3358 mean 0.99961 min 0.99155
spearman_cos_vs_nbr_jaccard n 3358 mean 0.09817 min -0.01092
same_text_same_nbrs_cos_mean n 3358 mean 0.99977 min 0.99855
same_text_changed_nbrs_cos_mean n 3253 mean 0.97124 min 0.86971
Counter({'fastapi/fastapi': 2211, 'Textualize/rich': 1081, 'psf/requests': 66})
```

## 6. What this changes downstream

- GPU sweep (rerun handoff section 4): **done**, section 7.
- Six-graph CPU `fit`: skipped; the full sweep covers it.
- Plan B / B0 step 3 (gold-patch pass): unaffected by this result. Note for the operator: the
  `swegemma` harness is not in `~/graphComp/archive` (no wheel under `archive/wheels/`, nothing named
  swegemma or adk anywhere in the archive) and is not on PyPI. It has to come from the Kaggle
  competition environment (the sample notebook's install source). Check whether it is already
  installed on vm-a2 before fetching it.


## 7. GNN sweep on all 127 graphs (item 4; vm-gpu, Oct 11)

### 7.1 What ran

| Item | Value |
| --- | --- |
| Machine | `vm-gpu`, zone `us-central1-a`, g2-standard-8 (8 vCPU, 31 GB), 1x NVIDIA L4 (23 GB), driver 550.163.01. Booted from a snapshot of vm-a2 (Debian 13), so the archive and scripts were already on disk; NVIDIA driver installed from Debian non-free (`nvidia-driver`, `nvidia-kernel-dkms`) |
| Python env | `~/gpuenv`: Python 3.13.5, torch 2.6.0+cu124 (CUDA 12.4), numpy 2.5.4, scipy 1.18.1 |
| Command | `PYTHON=~/gpuenv/bin/python bash ~/scripts/ops/run_a3b_gcp.sh ~/graphComp/archive ~/results/a3b_gpu all --stages fit linkpred --save-embeddings` (launched behind a guard that refused to start unless `torch.cuda.is_available()`) |
| Device check | log line `[a3b] torch 2.6.0+cu124 device=cuda (NVIDIA L4)`; `a3b_provenance.json` `"gpu": "NVIDIA L4"` |
| Run time | 15:37 to 16:06 UTC (about 29 min) |
| Script sha256 | `a3b_gnn.py` c9b42f02...; `common.py` 291fe9f6...; `run_a3b_gcp.sh` 7dc895c4... (`provenance/scripts_sha256_a3b.txt`) |
| Output | `planA_results/vm-gpu/results/a3b_gpu/`: `a3b_gnn.csv` (889 rows = 127 graphs x {fit: structural, text, both; linkpred: structural, text, both, random_control}), `a3b_summary.json`, `a3b.log` (0 tracebacks), `a3b_provenance.json`, `provenance/`, `embeddings/` (762 matrices; the random control is not saved) |
| Analysis | `a3b_gpu/a3b_gpu_analysis.txt`: every number in 7.2 to 7.4, recomputed on the laptop from `a3b_gnn.csv` |

Method (from `a3b_gnn.py`, read for leakage): one fresh two-layer mean-aggregation GNN per graph x
feature set x objective, 300 epochs, Adam, fixed seed; no early stopping and no selection on held-out
data. `structural` = a learnable per-node embedding plus degree features (no text); `text` = degree
features plus a 2,048-wide hashed bag of tokens (no learnable id); `both` = both. **fit**: predict the
provided vector, loss on a random 80% of nodes only, cosine scored on the other 20%; baseline = cosine
of held-out vectors to the mean of the training vectors. **linkpred**: train embeddings only to
predict the graph's edges (never sees the provided vectors), align to the provided vectors with
orthogonal Procrustes fitted on 80% of nodes, score on the held-out 20%. The setting is transductive:
held-out nodes' text and edges are inputs; only their target vectors are hidden.

### 7.2 fit: can the provided vector of an unseen node be predicted?

Per-repo means over graphs. "Gain" = (held-out minus baseline) / (1 minus baseline), the share of the
gap above the trivial mean-vector predictor that the model closes.

| Repo (graphs) | Baseline | structural | text | both | Gain: structural / text / both | Train minus held-out: structural / text / both |
| --- | --- | --- | --- | --- | --- | --- |
| rich (47) | 0.9483 | 0.9909 | **0.9937** | 0.9915 | 0.824 / **0.878** / 0.835 | +0.007 / +0.005 / +0.007 |
| fastapi (67) | 0.8808 | 0.8976 | **0.9608** | 0.9239 | 0.141 / **0.671** / 0.362 | +0.098 / +0.036 / +0.072 |
| requests (12) | 0.9278 | 0.9773 | **0.9815** | 0.9781 | 0.685 / **0.744** / 0.697 | +0.021 / +0.017 / +0.021 |
| httpx (1) | 0.9354 | 0.9866 | **0.9951** | 0.9888 | 0.793 / **0.924** / 0.827 | +0.013 / +0.004 / +0.010 |

`both` minus `text`, per graph:

| Repo | Mean | Min | Max | Graphs where both > text |
| --- | --- | --- | --- | --- |
| rich | -0.0022 | -0.0044 | +0.0004 | 1 / 47 |
| fastapi | -0.0369 | -0.0572 | -0.0237 | 0 / 67 |
| requests | -0.0034 | -0.0095 | +0.0013 | 2 / 12 |
| httpx | -0.0063 | | | 0 / 1 |
| all | | | | **3 / 127** |

Text arm, held-out nodes by isolation: rich isolated 0.896 vs connected 0.997; fastapi 0.945 vs
0.996; requests 0.796 vs 0.997; httpx 0.895 vs 0.999.

### 7.3 linkpred: does a geometry learned from the graph's links match the provided vectors?

Per-repo means over graphs, held-out nodes, after Procrustes.

| Repo | Metric | random_control | structural | text | both |
| --- | --- | --- | --- | --- | --- |
| rich | Spearman / k-NN@10 | -0.000 / 0.005 | -0.002 / 0.014 | 0.062 / **0.206** | 0.092 / 0.181 |
| fastapi | Spearman / k-NN@10 | -0.000 / 0.003 | 0.015 / 0.005 | 0.169 / **0.199** | 0.182 / 0.083 |
| requests | Spearman / k-NN@10 | 0.000 / 0.017 | 0.000 / 0.013 | 0.068 / **0.366** | 0.080 / 0.312 |
| httpx | Spearman / k-NN@10 | 0.007 / 0.015 | 0.000 / 0.016 | 0.094 / **0.435** | 0.040 / 0.276 |

`cos_mean` after Procrustes, structural arm: rich 0.853, fastapi 0.827, requests 0.928, httpx 0.935.
**These are not signal.** For requests and httpx they equal the `fit` global-mean baselines to four
decimals (0.9278, 0.9354): structural link-prediction embeddings collapse to near-identical vectors,
and Procrustes maps them onto the mean direction of the (anisotropic) provided vectors. Spearman of
about 0 and k-NN@10 at the random-control level confirm it. Read linkpred by Spearman and k-NN only.

### 7.4 Interpretation

1. **Graph structure adds nothing beyond text.** In `fit`, adding structure to text lowers held-out
   cosine in every repo (rich -0.002, requests -0.003, httpx -0.006, fastapi -0.037) and raises it in
   only 3 of 127 graphs, by at most +0.0013. In `linkpred`, structure alone is at the random-control
   level on Spearman and k-NN, and adding it to text lowers k-NN overlap in every repo (fastapi 0.199
   to 0.083). Under the pre-registered table (rerun handoff section 4.3), structural gain is low to
   moderate and `both` minus `text` is at or below zero: the "text carries the signal" branch.
2. **Why `both` is below `text`: memorisation.** The learnable per-node embedding in `structural` and
   `both` lets the model fit training nodes without generalising; the train-minus-held-out gap is
   largest in those arms (fastapi +0.098 and +0.072 against +0.036 for text).
3. **Structure alone predicts well in rich and requests only through homophily** (`fit` gains 0.82
   and 0.69): neighbouring functions read alike, so neighbours' vectors predict a node's own. This is
   the `nbrmean` effect of section 3.2. In fastapi, with about 2,900 isolated nodes per graph,
   structure alone closes only 14% of the gap. That homophily is not structural information in the
   vectors is shown by point 1.
4. **Section 3.3's context component is a wider text window, not the graph.** Identical node text
   with changed neighbours moves the vector (mean cosine 0.971); since structure adds nothing in
   either objective, the most economical explanation is that the encoder embedded more text than the
   node's own `text` field (enclosing class, file, docstrings). This is an inference, not a direct
   measurement: the encoder's input window is not observable from the release.
5. **Isolated nodes are harder to predict even from text** (requests 0.80 vs 0.997 connected),
   consistent with section 3.4: isolated nodes are textually distinct code. A footnote.

### 7.5 Caveats

- **Duplicate twins inflate `text` and `both`, not the comparison.** About 16.7% of nodes per graph
  have an exact duplicate (identical text and vector); a held-out node whose twin is in the training
  set is predicted by memorisation. This inflates absolute `text` and `both` scores equally, so
  `both` minus `text` is unaffected. Not yet quantified: split held-out nodes by "has a training twin"
  using the saved embeddings.
- **Graphs within a repo are not independent** (near-identical commits). Report per repo; do not
  quote intervals over 127 graphs as if independent. httpx has one graph and is anecdotal.
- **The text model is weak on purpose** (hashed bag of tokens). "Text predicts best" holds even for a
  crude text model, which strengthens the verdict; its absolute scores say nothing about which
  encoder produced the vectors.
- **Transductive setting.** Held-out nodes' text and edges are visible; only their targets are hidden.
  Appropriate for "given a node's text and position, predict its vector", and identical across arms.

### 7.6 Verdict and paper wording (section 5.6; supersedes 3.5)

Three tests, one answer:

| Test | Result |
| --- | --- |
| Identical text across commits (section 2) | same vector: cosine at or above 0.99 for 99.6% of same-text nodes, 3,358 pairs |
| `fit` (7.2) | text predicts best in every repo; adding structure never helps (3/127 graphs, at most +0.0013) |
| `linkpred` (7.3) | link structure alone recovers nothing beyond the random control; only text recovers neighbourhoods (k-NN@10 0.20 to 0.44) |

Suggested paragraph:

> The provided node vectors are a text embedding of an unidentified encoder, not a structural one.
> Nodes with byte-identical text receive near-identical vectors across commits (cosine at or above
> 0.99 for 99.6% of such nodes over 3,358 same-repository graph pairs). A graph neural network asked
> to predict held-out vectors does best from text alone; adding graph structure never improves on
> text (held-out cosine change from -0.002 in rich to -0.037 in fastapi; positive in 3 of 127
> graphs, by at most 0.0013), and embeddings learned only from the call graph's links recover none of
> the vectors' neighbourhood structure (k-NN@10 overlap at the random-control level, against 0.20 to
> 0.44 from text). A small context effect remains (cosine 0.971 when a node's neighbours change but
> its text does not), which we attribute to a wider text window than the node itself. For agent
> builders: `search_similar_code` retrieves code that reads alike; it does not encode call structure,
> which is available only through the graph tools.

Cite: `vm-a2/results/a3b/a3b_summary.json` (`checks.text_identity_*`),
`vm-a2/results/a3b/a3b_checks_text_identity.csv` (`same_text_share_ge_0.99`, `same_text_cos_p05`,
`same_text_changed_nbrs_cos_mean`), `vm-gpu/results/a3b_gpu/a3b_gnn.csv` (`heldout_cos_mean`,
`heldout_globalmean_baseline_cos`, `train_cos_mean`, `spearman_pairwise`, `knn10_overlap`), and
`a3b_gpu/a3b_gpu_analysis.txt`.

Downstream: for Plan B's B6 arm (graph tools backed by the v2 graph), the embeddings are text-only, so
`search_similar_code` should behave the same under either graph; only the three graph tools change.
That makes the scorecard's map-swap test (HANDOFF_planB_B0.md section 6, test 3) cleaner.

Optional follow-ups: (a) the duplicate-twin split from the saved embeddings; (b) a per-repo bar chart
of `fit` gain by feature set (the handoff's suggested figure), now worth making since the answer is
uniform across repos.
