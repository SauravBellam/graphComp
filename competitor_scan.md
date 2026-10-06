# Competitor scan: Gemma 4 Developer Agent, paper track

Snapshot taken 2026-10-06. Paper deadline is Nov 12, so nobody has "won" yet. This is what is publicly visible.

## Bottom line

- **No paper-track ranking exists yet.** Judging happens after Nov 12, and Kaggle Writeups are usually not browsable until the deadline. Kaggle's own pages (discussion, writeups, code) render only with JavaScript and could not be read from here, so this scan covers GitHub, the community leaderboard tracker, and web search.
- **Three public paper-track repos found.** Only one is rigorous and credible (yosh95). One makes headline claims that do not match the leaderboard (GDPR-SWE). One is a thin graph-agent writeup (GraphSWE-Gemma).
- **The prediction leaderboard is the best public signal of what works:** top score is 0.24 (Yurnero), then a cluster at 0.17–0.18. Most of the 1,917 teams sit at 0.08–0.13. The hidden board is 58 tasks, so 0.24 ≈ 14 tasks solved and one task ≈ 1.7 points.

## Prediction leaderboard (proxy for "what works")

Source: community tracker [kawchar85/gemma4-agent-leaderboard](https://github.com/kawchar85/gemma4-agent-leaderboard) (polls Kaggle every 5 min; data updated 2026-10-06 15:22 UTC). Official board: [kaggle.com/competitions/gemma-4-developer-agent/leaderboard](https://www.kaggle.com/competitions/gemma-4-developer-agent/leaderboard).

| Rank | Team | Public score |
|---|---|---|
| 1 | Yurnero | 0.24 |
| 2–4 | Shan Yang, Wilmer E. Henao, Tsunato Nakai | 0.18 |
| 5–16 | AGENTS.md, Kirill Labzin, Haoyu Wang, … (12 teams) | 0.17 |
| — | 61 teams | 0.15 |
| — | ~940 teams | 0.10–0.13 |

None of the top-15 teams has a public repo or writeup that I could find. Late-September notes in several repos say "top public 0.15; most good prompt-only configs score 0.10–0.12", so the leaders moved up roughly 0.06–0.09 in the past week.

## Paper-track entries found

### 1. "Every Tool Call Is a Token You Cannot Buy Back" (yosh95), the strongest so far
Repo: [yosh95/g4-developer-agent-paper](https://github.com/yosh95/g4-developer-agent-paper). README says it was submitted to the paper track. The draft is ~2,100 words.

- **Angle:** cost accounting and falsification, not a new method. "Nine design rules we measured — and mostly falsified."
- **Setup:** Gemma 4 31B INT4 on a 4-core CPU box, with the official harness replayed offline. Validated with gold patches (18/18 resolve).
- **Findings:**
  - *Spinning:* 20 of 55 tasks contain a run of 10 or more byte-identical tool calls, the longest is 74, and up to 34.6% of the budget goes to one chain. A stop-loss prompt cut calls by 29% with no accuracy change.
  - *Delegation:* a same-model sub-agent hurt (12/18 vs 6/18, McNemar p=0.07).
  - *Budget:* doubling it did nothing (11/11/11).
  - *Self-verify:* a "verify before submit" rule produced empty patches.
  - *Per repo:* success varies a lot (requests 7/8, fastapi 10/37, rich 2/9).
  - *Public vs hidden:* a public task is worth about 1/5 of a hidden task (34.5% local became 5.2% hidden).
  - *Localization vs resolution:* the graph/lexical localizer finds the gold file in the top 5 for 14/18 tasks, yet those tasks still fail. Finding the right file is not the bottleneck.
- **Artifacts:** per-task CSV, spin statistics, offline replay harness, Apache-2.0. Scores high on verifiability.
- **Weakness:** no new method, n=18 paired comparisons, and it uses only a little of the graph/embedding dataset.

### 2. GDPR-SWE: Graph-Guided Dual-Process Reasoning (manojvenaram)
Repo: [manojvenaram/GDPR-SWE-Gemma-4-Developer-Agent](https://github.com/manojvenaram/GDPR-SWE-Gemma-4-Developer-Agent)

- **Method:** a "System 1" AST-graph navigator sub-agent plus a "System 2" Gemma executive, with a five-phase loop that must reproduce the bug before patching. Includes QLoRA and synthetic-trajectory scripts.
- **Claims:** 55% pass@1 on 129 tasks, vs ReAct 28.7% and Agentless 31.8%, plus a 68.6% token reduction.
- **Red flags (my inference, not verified by running it):**
  - Results come from a `mock_swegemma.py` emulator, not the real harness.
  - It lists SymPy and Pydantic tasks, which are not in the competition's 129 tasks (fastapi, rich, requests, httpx).
  - 55% is far above anything on the board, where the best is 0.24 hidden and about 35% on public tasks per yosh95.
  - Judges scoring verifiability will likely notice.

### 3. GraphSWE-Gemma (nazarcoder123)
Repo: [nazarcoder123/…Paper-Track-Kaggle-](https://github.com/nazarcoder123/Google---The-Gemma-4-Developer-Agent-Paper-Track-Kaggle-)

- **Method:** NetworkX AST multigraph, 256-d symbol embeddings, and caller/callee subgraph "context compaction".
- **Claims:** 93.1% token compression, evaluated on mock data only. No resolution numbers.

### Other related repos (agent track, show the common playbook)
- [msingatullin/gemma-4-developer-agent](https://github.com/msingatullin/gemma-4-developer-agent): plans both tracks. Paper angle is a "modular verification framework + structured code-graph search". Scaffolding only.
- [Gosling-dude/gemma4-developer-agent](https://github.com/Gosling-dude/gemma4-developer-agent): single coder agent plus 3 skills. Graph tools are used only when the cause spans functions. On a small ablation (n=12), graph localization recall@1 was 0.50 vs 0.08 for grep. No real Gemma runs yet.
- [hemantadil/kaggle_gemma4_dev_agent](https://github.com/hemantadil/kaggle_gemma4_dev_agent): flat tools instead of sub-agents, temperature 0.05–0.1, thinking budget of 2048+ tokens, edits of 1–10 lines.
- [bansal1600/kaggle-finetune-gemma](https://github.com/bansal1600/kaggle-finetune-gemma): prompts first, then LoRA on successful trajectories, with GRPO as a stretch goal.
- [ahmad-alalony-ramaaz/gemma-4-agent-lab](https://github.com/ahmad-alalony-ramaaz/gemma-4-agent-lab): analyzer+coder pipeline, single-variable ablations. Based on Aleksei Provorov's starter notebook.
- [rishaviitd/kaggle.gemma.coding.agent](https://github.com/rishaviitd/kaggle.gemma.coding.agent), [apedintensor/gemma-developer-agent](https://github.com/apedintensor/gemma-developer-agent), [ennydev-2026](https://github.com/ennydev-2026/gemma-4-developer-agent), [8kta/kaggle-gemma4](https://github.com/8kta/kaggle-gemma4), [sweeden-ttu/developer-agent](https://github.com/sweeden-ttu/developer-agent): scaffolds and notes.
- Kaggle notebooks (not readable from here): [Getting Started (organizer)](https://www.kaggle.com/code/ryanholbrook/getting-started-gemma-4-developer-agent), [AST Graph-Augmented SWE Multi-Agent System](https://www.kaggle.com/code/avikdas567/gemma-4-ast-graph-augmented-swe-multi-agent-system).

## General strategy and direction

1. **Graph-guided localization is the default paper pitch.** Most paper repos are "AST/code graph + embeddings → localize → compress context → patch". This space is crowded and mostly unvalidated. Claims are token-reduction percentages or mock-harness pass rates.
2. **The credible leader is going empirical and contrarian.** yosh95 reports honest negative results, runs paired statistics, releases a replay harness, and focuses on cost and failure modes. That scores well on verifiability and novelty-of-insight without needing a new model.
3. **On the agent side, simple beats elaborate so far:**
   - single agent and flat tools
   - low temperature
   - small surgical edits
   - targeted tests
   - anti-loop and stop-loss rules

   Multi-agent delegation is measured as harmful. Fine-tuning (LoRA on trajectories, GRPO) is planned by many but has no public result yet.
