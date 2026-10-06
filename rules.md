# Gemma 4 Developer Agent Competition: Rules & Reference

Merged from `Comp. Overview.pdf`, `Dataset Description.pdf`, and `Proj-Outlines.pdf`. Each fact appears once.

> **Project focus: this project is entered to win the Paper Track.** The goal is a top-3 paper by the **Nov 12, 2026** deadline, judged on the rubric in section 5. The prediction competition (sections 6–7) matters only as context and as a possible source of experiments for the paper.

## 1. The competition at a glance

**Goal:** post-train an open model (Gemma 4) into a reliable agent that navigates complex Python codebases and drafts fixes for real bugs and feature requests, running offline on consumer-class hardware. Fine-tuning and reinforcement learning are the expected levers.

There are **two separate tracks**; you may enter either or both:

| | Prediction competition | Paper track |
|---|---|---|
| What you submit | `submission.zip` agent config, scored automatically | A Kaggle Writeup (research paper) |
| Ranked by | % of hidden test tasks fixed | Judges' rubric (section 5) |
| Start | Sept 23, 2026 | Sept 22, 2026 |
| Final deadline | Dec 2, 2026 | **Nov 12, 2026** |
| Prize pool | $65,000 | $35,000 |

Entering the main competition is **not** required for the paper track, though both are encouraged. The paper may document your approach to the prediction competition or be any related unpublished research.

## 2. Timeline

All deadlines are **11:59 PM UTC** unless noted. Organizers may change the timeline.

| Date (2026) | Event |
|---|---|
| Sept 22 | Paper track opens |
| Sept 23 | Prediction competition opens |
| **Nov 12** | **Paper track final submission deadline** |
| Nov 25 | Entry deadline (accept the rules to compete) and team merger deadline |
| Dec 2 | Prediction competition final submission deadline |

## 3. Prizes ($100,000 total)

- **Prediction competition ($65,000):** 1st $37,000 · 2nd $18,000 · 3rd $10,000
- **Paper track ($35,000):** awarded to the three highest-scoring papers. Top papers are also highlighted at a Google-hosted event (e.g., a NeurIPS expo workshop).

## 4. Paper track

### 4.1 Eligibility
Join the paper-track competition and submit a Kaggle Writeup documenting research that advances the state of the art in agentic software engineering. The research must be **original and unpublished**.

### 4.2 Required content (Kaggle Writeup)
- Title and subtitle
- Abstract
- Introduction
- Description of the research, including methods and experiments
- Related work and citations

### 4.3 Format limits and options
- **Length:** 3,000 words maximum.
- **Non-archival:** submitting here does not block submission to other conferences or venues.
- **Optional public notebook:** code goes in the Project Links field as a public notebook (no login or paywall). Private Kaggle notebooks become public automatically after the deadline.
- **Optional PDF:** instead of writing in the Writeup editor, you may upload an arXiv-ready (LaTeX/Word) PDF via the Public Project Link feature (must be public, no login or paywall).
- Any private Kaggle Resource attached to the public Writeup becomes public after the deadline.

### 4.4 How to submit
Click "New Writeup", save it, then click "Submit" (top right). **Unsubmitted or draft Writeups at the deadline are not judged.**

### 4.5 Suggested topics (not exhaustive)
- **Tuning & Optimization:** parameter-efficient fine-tuning and RL approaches for SWE agents.
- **Code Comprehension:** new techniques for code-graph generation, parsing, and embedding.
- **Tasks & Benchmarks:** new tasks, evaluation datasets, or resources targeting code structure, graphs, and structured code generation.
- **Graph Reasoning:** methods and architectures for LLM reasoning over large-scale graphs.

Papers that use the competition's released graph and embedding dataset are **highly encouraged** but not required.

## 5. Judging the paper

Five criteria, each scored 0 to 5, equally weighted; final score is the average.

| Criterion | What judges ask |
|---|---|
| **Novelty** | Does it give new insights, deepen understanding, or highlight important properties of existing methods? |
| **Quality** | How general is the approach beyond this competition? Does it transfer to similar problems? |
| **Relevance** | Significance and potential impact on software engineering and agentic learning. |
| **Verifiability** | Enough detail to understand how the method works and how data was obtained, analyzed, and interpreted. |
| **Clarity** | Is it clearly presented and written? |

- Rubric scores are **not shared** with entrants.
- **Ties** go to the paper entered first.
- If a potential winner is disqualified, the next-highest score moves up.

## 6. Prediction competition: submission format

Submit `submission.zip` with `agent.yaml` at the root. It is compiled into Google ADK agents and follows the **Google ADK Agent Config** spec, with extra restrictions to prevent code execution outside the sandbox.

```
submission.zip
├── agent.yaml                 # REQUIRED root agent config
├── configs/sampling.yaml      # optional generation params (via !include)
├── prompts/system.md, ...     # optional system instructions (via !include)
├── sub_agents/*.yaml          # optional sub-agent / AgentTool configs
├── adapters/<name>/           # optional PEFT LoRA adapters
│   ├── adapter_config.json
│   └── adapter_model.safetensors
└── skills/<skill>/            # optional ADK skill directories
    ├── SKILL.md
    ├── scripts/               # Python/bash run in the sandbox
    └── resources/             # domain-knowledge markdown
```

### 6.1 Model and adapters
- The only supported model is **`gemma-4-31b-it-qat-w4a16-ct`**, and every agent and sub-agent must use it.
- LoRA adapters are optional and must be `.safetensors`. Put each in `adapters/<adapter_name>/` and reference it with `adapter: <adapter_name>` on any `LlmAgent` (in `agent.yaml` or `sub_agents/*.yaml`). Different agents may use different adapters (e.g., `main_lora` on the root coder, `tool_lora` on a read-only analyzer).

### 6.2 Tool, prompt, and skill rules
- `!include` paths resolve relative to the file containing the tag.
- No path traversal outside the submission root (no `../`, no symlinks).
- Agents may only call harness-provided tools or custom sub-agents defined via `agent_tool`.
- Each skill is a directory with a `SKILL.md` whose YAML frontmatter has `name: <skill-name>`. Scripts run via `run_skill_script` inside the persistent Docker container, share the filesystem with `run_command`, and count against the time budget. Resource files are read with `load_skill_resource`.

### 6.3 Harness tools
| Tool | Purpose |
|---|---|
| `run_command(command)` | Run a bash command in `/workspace` |
| `submit_patch()` | `git add -N .` then capture `git diff HEAD` |
| `get_status()` | Live budget use and patch status |
| `read_file(filepath, start_line?, end_line?)` | Read a file, 1-indexed inclusive lines |
| `edit_file(filepath, old_string, new_string, allow_multiple=False)` | String replace in an existing non-empty file |
| `write_file(filepath, content)` | Create/overwrite a file, auto-creating parent dirs |
| `get_code_neighbors(node, edge_type?, max_neighbors=50)` | Incoming/outgoing neighbors of a symbol in the call/dependency graph |
| `search_similar_code(query, k=10)` | Top-k graph nodes by cosine similarity over precomputed embeddings |
| `get_code_subgraph(nodes)` | Induced subgraph over a list of symbols |

See `HARNESS_README.md` in the dataset for full details.

## 7. Prediction competition: scoring and compute

- **Metric:** SWE-bench-style PASS/FAIL. Each patch is applied to its repo and that task's validation tests are run. Score = % of tasks whose tests pass.
- **Time limit:** 12 hours for all tasks, including sandbox setup, excluding patch validation. Optional per-task limits can be set in `eval_config.yaml`.
- **Output:** `/kaggle/working/submission.parquet` with columns `id` (the `instance_id`) and `prediction` (unified diff vs. `base_commit`, or `NO_PATCH`).
- **Hardware:** L4x4 machines (96 GB GPU memory) are available. They use GPU quota at 2x the T4x2/P100 rate (may increase), are restricted to notebooks attached to this competition (circumvention can lead to team or account bans), and must run with **internet disabled**.

## 8. Dataset

The public dataset is the **training set only**; at scoring time it is swapped for a hidden test set built by the same pipeline with identical graph generation and verification.

### 8.1 Scale and sources
- **Training:** 129 tasks from `fastapi/fastapi`, `Textualize/rich`, `psf/requests`, `encode/httpx`.
- **Test:** about 120 tasks, split evenly between public and private leaderboards, curated from **private repositories**.

### 8.2 Files
- **`tasks.jsonl`**: one task per line with `instance_id` (e.g., `fastapi_11194`), `repo`, `base_commit` (40-char SHA), `problem_statement` (the agent's prompt), `hints_text` (may be empty), `patch` (reference fix; absent from the hidden test set), `test_patch` (pytest verification tests; kept private during scoring), and `created_at` (ISO-8601).
- **`snapshots/<instance_id>.tgz`**: full git repo frozen at `base_commit` with forward history removed, so `git status/log/diff` are safe.
- **`graphs/`**: NetworkX AST call/dependency graphs as JSON (256 files: 129 per-task names hard-linked to 127 `<repo_short>_<base_commit>.json` files). Directed multigraph. Nodes have `id`/`name` (fully qualified symbol, e.g. `fastapi.routing._prepare_response_content`) and `text` (full source at `base_commit`). Edges have `source`, `target`, `type` (e.g. `calls`), and `key` (index for parallel edges). Backs `get_code_neighbors` and `get_code_subgraph`.
- **`embeddings/`**: `.npz` archives (same 256-file layout); each `<node_id>.npy` is a float32 vector of length 256. Backs `search_similar_code`.
- **`wheels/`**: 124 offline Python wheels (fastapi, starlette, pydantic, requests, urllib3, rich, httpx, httpcore, pytest, and transitive deps), mounted read-only at `/wheels/`.
- **`docker/`**: `Dockerfile.sandbox` (Python 3.13, git, pytest, setuptools/hatchling/flit-core/poetry-core/pdm-backend), `Dockerfile.public` (dev image for the 129 tasks), and `imp.py` / `telnetlib.py` shims for modules removed in Python 3.13.
- **`sandbox/setup.py`**: run in `/workspace`; reads `pyproject.toml` / `setup.cfg` / `requirements.txt`, does `pip install --no-index --find-links=/wheels -e .`, and makes a clean baseline commit so `git diff HEAD` shows only agent edits.
- **`sample_submission/`**: runnable baseline showing the YAML schema, multi-agent delegation, and multi-LoRA routing.
- **`HARNESS_README.md`**: guide to the `swegemma`, `adk-submission`, and `adk-eval-core` libraries, the two-container sandbox lifecycle, tool signatures, context compaction, and local evaluation CLI.

### 8.3 How tasks were built and verified
1. **Mining:** commits that change both core `.py` logic and matching unit tests (`test_*.py` / `*_test.py`).
2. **Issue linking:** each change matched to its original issue or bug report.
3. **Filtering:** dropped mechanical refactors, docs-only changes, large automated changes, and ambiguous issues without reproducible acceptance criteria.
4. **Execution verification:**
   - *Fail-to-pass:* `test_patch` on `base_commit` without the fix must fail.
   - *Pass-to-pass:* fix plus `test_patch` must pass the whole suite.
   - *Frontier check (test set only):* a larger frontier model must pass, or come within one test case of passing.
