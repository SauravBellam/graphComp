# Literature Review: Code Graphs, Training, and Evaluation for Open SWE Agents

*Prepared 2026-10-06 for the Gemma 4 Developer Agent Competition paper track (deadline Nov 12, 2026; 3,000 words). This is a neutral survey of the literature; it deliberately contains no gap analysis or recommended study. Sources found with alphaXiv and Consensus; every claim below is tied to a numbered reference at the end. Where only an abstract was read, the text says so or limits itself to what the abstract states.*

---

## 0. How this review is organized

1. **Broad survey (Part A):** the state of AI software-engineering agents as a whole: benchmarks and their problems, agent scaffolds, training open models, and practical constraints (small/quantized models, context limits).
2. **Focused survey (Part B):** the four competition topics, weighted toward what the released dataset supports (AST call/dependency graphs + 256‑d node embeddings + the `get_code_neighbors` / `search_similar_code` / `get_code_subgraph` tools): code graphs for agents, graph-integrated models, graph reasoning by LLMs, and RL/PEFT for SWE agents.

**Competition context.** The target model is Gemma‑4‑31B (4‑bit weight-quantized, `w4a16`), offline, on L4×4 hardware, with a 12‑hour budget for ~120 hidden tasks from *private* repositories. Training data is 129 tasks from four public Python repos (fastapi, rich, requests, httpx). Graph edges are typed (e.g. `calls`), nodes carry full source text, and embeddings are precomputed (see `rules.md`).

---

## Part A. Broad survey of the field

### A1. The task and its benchmarks

**SWE-bench** framed the task: given a repository snapshot and a GitHub issue, produce a patch that makes held-out tests pass; at release the best model solved 1.96% of 2,294 issues [1]. It spawned variants (Lite, Verified, Java [2], Live [3], Pro [4], rebench [5]) and became the standard metric.

**The benchmark is noisy and partly contaminated.** Three lines of evidence:
- *Solution leakage and weak tests.* SWE-Bench+ found 32.67% of SWE-Agent+GPT-4 "successes" had the solution in the issue text and 31.08% passed because of weak tests; filtering cut the resolve rate from 12.47% to 3.97% [6]. A 2026 follow-up reports 60.83% of commonly-resolved Lite/Verified issues contain leakage and a test-strengthening tool lowers resolve rates by 27–36 points [7].
- *Memorization.* Proprietary frontier models localized edited files ~6× better on SWE-bench Verified than on comparable fresher benchmarks *from the issue text alone*, which suggests recall rather than reasoning [8]. SWE-rebench [5] and SWE-bench-Live [3] report the same direction (scores drop on fresh tasks).
- *Leaderboard saturation.* A 2026 audit of 254 submissions argues the SWE-bench leaderboard can no longer order its top entries [9] (abstract only).

**Responses.** Fresh/live benchmarks [3][5], private or held-out sets (SWE-Bench Pro's commercial split [4], SWE-bench-secret [10]), and from-scratch tasks with hand-written verifiers (DeepSWE, where an LLM judge disagrees with its verifier 1.4% of the time vs 32.4% for SWE-Bench Pro's inherited tests [11]). Beyond pass/fail, new work measures exploration (SWE-Explore [12]), trajectory structure ("What Resolve Rate Hides" [13]), and non-functional quality such as policy compliance (SWE-CC [14]) and review quality (SWE-Gate [15]) (all abstract only).

**Competition test set.** The competition's hidden test set comes from private repos, uses fail-to-pass/pass-to-pass verification, and adds a frontier-model solvability filter (`rules.md` §8.3).

### A2. Agent scaffolds

- **Agent-computer interfaces.** SWE-agent showed that interface design matters: a purpose-built ACI improved pass@1 to 12.5% on SWE-bench [16].
- **Structure-aware agents.** AutoCodeRover searches over AST-level classes/methods plus spectrum-based fault localization (19% on Lite, ~$0.43/issue) [17]; SpecRover adds specification inference (+50% over AutoCodeRover) [18].
- **"Agentless" pipelines.** A fixed localize → repair → validate pipeline reached 32% on Lite at $0.70, beating contemporaneous agents and becoming a common baseline [19][20].
- **Multi-agent.** MAGIS (manager/custodian/developer/QA, 13.94% on early SWE-bench) [21]; SGAgent (localize → suggest → fix with a repository knowledge-graph toolkit, 51.3% on Lite with a proprietary LLM) [22].
- **Refinement and review.** REFINE turns near-correct "draft patches" into correct ones (+14.67% for AutoCodeRover) [23]; SWE-Review closes the loop with agentic code review [24] (abstract only).
- **What agents actually do.** A study of 120 trajectories from RepairAgent, AutoCodeRover and OpenHands identifies behavioral motifs and anti-patterns that separate success from failure [25].

**Tool *architecture* vs tool *capability*.** The most controlled study to date (COLM 2026, 11,700 trajectories, 3 actors including Qwen3‑Coder‑30B) holds capability fixed and varies only how tools are exposed. Results: structured "atomic" tools improve repeat-run consistency by up to 4.7×, **with the largest gains for the weakest actor**; NL search widens exploration by >11%; CodeAct-style Python cuts steps 41.6% and tokens 56.3%; scratchpad/hypothesis tools barely change behavior [26]. The study excluded embedding and graph capabilities to isolate architecture [26].

### A3. Training open-weight models to be SWE agents

**Environments and data.** SWE-Gym (2,438 executable Python tasks; trains agents *and* verifiers) [27]; R2E-Gym (procedural environments, hybrid verifiers) [28]; SWE-rebench (21k+ tasks for RL) [5]; SWE-smith (used by CodeScout for 39k training instances) [29]; CodeMidas and Prime Intellect's 365k environments scale this further [30][31] (abstracts only).

**SFT / distillation.** SWE-Fixer (BM25 + small model retrieval then edit; 32.8% Verified with filtering) [32]; SWE-Lego (pushing SFT alone) [33]; SWE-Master (reproducible post-training) [34]; NVIDIA's SWE-ZERO→SWE-HERO two-stage SFT [35] and Open-SWE-Traces [36] (abstracts only).

**RL.** SWE-RL used a rule-based patch-similarity reward on software-evolution data: Llama3‑SWE‑RL‑70B reached 41.0% on Verified and generalized to out-of-domain reasoning, while SFT degraded it [37]. Multi-turn RL (rejection fine-tuning, then DAPO) took Qwen2.5‑72B from 11% to 39% on Verified [38]. Agent-RLVR adds teacher-style guidance to fight sparse rewards (Qwen2.5‑72B 9.4%→22.4%) [39]. BoostAPR uses dual reward models with line-level credit (40.7% Verified) [40]. Execution-free reward models (SWE-RM, MoE 30B/3B-active) improve both test-time scaling and RL, and the authors note TTS accuracy does not predict RL usefulness; calibration matters [41]. A 2026 survey organizes reward design for SE by source, granularity and aggregation [42].

**Credit assignment.** SWE-MILE (milestone credit), TRACE (turn-level credit), "Dependency-Aware Reward Shaping", LEGO-RL (harness-native RL), and multi-harness RL all target the sparsity of terminal pass/fail [43][44][45][46][47] (abstracts only).

**Parameter-efficient tuning.** LoRA/QLoRA are standard (QLoRA fine-tunes 65B on one 48GB GPU via a frozen 4-bit base) [48]; DoRA improves LoRA's learning capacity [49]. Agent-specific PEFT: MoRAgent assigns separate LoRA groups to reasoner/executor/summarizer roles [50]. The competition's sample submission already supports **multi-LoRA routing** (`rules.md` §8.2).

### A4. Practical constraints: small, quantized, long-horizon

- **Small agents.** FrogNano trains a 4B coding agent via online task synthesis [51]; Mingbird is a local-first harness so 2–9B models can finish real tasks [52]; Kozuchi Agent targets open-weight industrial repair [53] (abstracts only).
- **Context management.** CompactionRL, AutoCompact (learning *when* to compact), SWE-Pruner Pro, and a UT Austin study of context compression [54][55][56][57] (abstracts only). The competition harness has its own compaction (`HARNESS_README.md`).
- **Quantization damages agents invisibly.** On τ²-bench, 4-bit AWQ looked lossless on task score, but **Gemma‑4‑31B's tool-name hallucination rate doubled (19.5%→38.3%) and failure volume grew 2.5×**; the benchmark's error budget hid it, and tightening the budget from 10 to 2 errors exposed a 16.7-point gap. Quantization *amplified existing failure modes* rather than creating new ones, and a reflexive repair prompt that re-injects the tool inventory fixed it (+5.9 points) [58]. A separate APR study finds similar hidden costs from quantization [59] (abstract only). The competition uses the same model family at 4-bit precision.

---

## Part B. Focused survey of the competition topics

### B1. Code Comprehension: code graphs given to agents

| System | Graph | How the agent uses it | Headline result |
|---|---|---|---|
| RepoGraph [60] | line/function-level repo graph | ego-graph retrieval plug-in | improves several frameworks on SWE-bench (abstract) |
| CodexGraph [61] | Neo4j code graph DB | agent writes Cypher queries | depends on LLM's query-writing ability (as summarized in [62]) |
| LocAgent [62] | heterogeneous graph: dir/file/class/function; contain/import/invoke/inherit | `SearchEntity`, `TraverseGraph` (typed multi-hop BFS), `RetrieveEntity` | 92.7% file-level acc; LoRA-SFT Qwen2.5‑7B ≈ a proprietary LLM at ~86% lower cost; tree-formatted subgraphs beat edge lists/DOT/JSON |
| CoSIL, OrcaLoca, RepoSearcher, RepoNavigator | on-the-fly call graphs / AST tools / LSP "jump" | localization tools | summarized in [63] |
| ARISE [64] | adds statement-level def-use (data-flow) edges | data-flow slicing tool on SWE-agent | Qwen2.5‑Coder‑32B: 22.0% Lite (+4.7 pts); Function R@1 0.43→0.60; ablation credits the data-flow graph, not the tool schema |
| SemNav [65] | AST graph + **language-server resolution on demand** | semantic cards, candidate workspace | Gemma‑4 4B: File Hit@10 68.3→82.7 on Lite; downstream resolve 44.0→52.3% (Qwen3.5‑35B repair) |
| DualView [66] | module coupling, call, class-hierarchy, PDG views | graphs returned as **images + text** | visual > textual graph serialization (78 vs 71 resolved in ablation); +46 instances on SWE-bench Pro for OpenCode/Kimi |
| LARGER [67] | AST graph, confidence-weighted edges | **no new tool**: graph neighbors injected into grep results | LocBench Acc@5 +11.8 over best baseline; removing graph expansion is the biggest drop (−13.5%) |
| CodeAnchor [68] | PyCG call/inheritance graph | **no new tool**: structure injected as source comments | +2.2 Func@5, −1.6 rounds, roughly halves run-to-run variance; dense tags saturate; hub-heavy repos need "who-calls-me"-only links |
| Codebase-Memory [69], RepoAtlas [70], Know-Before-Fix [71] | tree-sitter KG via MCP; multimodal repo views; QA-driven knowledge | various | abstracts only |

**Graph-based localization without an LLM agent.** GREPO (86 Python repos, 47,294 tasks) shows **GNNs (GAT/GATv2) on the repo graph beat LocAgent and Agentless** at function/class localization (e.g., ~31% vs 11–21% Hit@5), with embedding-retrieved anchor nodes seeding the subgraph and a scaling trend across training repos [72]. GraphLocator and SweRank+ are related ranking approaches [73][74] (abstracts only).

**Representation matters, not just retrieval.** Role-aware LLM summaries of files beat paths by up to 40% Hit@5 at 10–21× less footprint than raw source; combining representations helps; small models (7B) struggle to use summaries [75]. LocAgent likewise found tree-formatted subgraphs outperform other textual encodings [62].

**Older but foundational lines (Consensus).** Program graphs for neural bug detection and repair: PDG/DFG context [76], Hoppity's graph transformations [77], GINN interval GNNs [78], GLANCE graph-to-sequence repair [79], code property graphs for repair [80][81][82], and GNN code search (GraphSearchNet, DGMS, deGraphCS) [83][84][85].

**Python call-graph construction.** Static call-graph generation for Python is studied as its own problem: PyCG [105], application-centered construction [106], and bytecode-level analysis (CPyGraph) [107] (abstracts only).

### B2. Graph-integrated models (structure inside the network)

**Code Graph Model (CGM)** encodes each node's code with a CodeT5+ encoder, maps it through an adapter into the LLM as one token per 512-token chunk, and **replaces attention between node tokens with the graph adjacency mask**; trained with LoRA plus a graph-to-code reconstruction pretask. Qwen2.5‑72B CGM + an agentless Graph-RAG pipeline: 43.0% Lite, 50.4% Verified [86]. Ablations: removing the graph mask costs 5.6–8.6 EM on code completion; but **backbone size matters a lot** (Qwen2.5‑Coder‑7B CGM: 4.0% on Lite) [86]. Related: graph-aware LoRA generation [87], deep graph-language fusion [88], bridging code graphs and LLMs [89] (abstracts only). Mechanistic work warns that **graph tokens can become attention "sinks"** the LLM ignores [90] (abstract only).

### B3. Graph Reasoning by LLMs

- **Encoding matters.** "Talk like a Graph" showed LLM graph reasoning varies with encoding, task, and graph structure; the right encoder gains 4.8–61.8% [91]. NLGraph found LLMs show preliminary graph reasoning that degrades on complex problems and is brittle to spurious correlations [92]; GPT4Graph reached similar conclusions [93].
- **Agentic graph exploration.** Graph-CoT has the LLM iteratively reason, query the graph, and execute [94]. 2026 work trains graph agents (GraphCert), studies agentic graph-token reasoning, and treats knowledge graphs like a filesystem with `ls`/`cat`-style tools [95][96][97] (abstracts only). GTA benchmarks multi-step algorithmic graph reasoning [98].
- **Surveys.** LLMs on graphs [99], Graph2text vs Graph2token [100], graph ML in the LLM era [101].

These works mostly use citation, product and knowledge graphs.

### B4. Tuning & Optimization for graph-equipped agents

- **SFT on graph-tool trajectories** works: LocAgent's LoRA SFT on 768 proprietary-model/self trajectories made Qwen2.5‑7B competitive [62].
- **RL with a plain terminal can match graph tools.** CodeScout trains Qwen3 1.7B/4B/14B with GSPO on a **bash-only** scaffold, reward = file+module+function F1 vs the gold patch; it matches or beats 2–18× larger models and graph-tool agents. During RL the policy **collapses to just `rg` and `sed`** [63]. Better localization from CodeScout‑14B lifts Qwen3‑4B's resolve rate 13.4→17.2%; oracle locations give 19.6% [63].
- **Action-level credit with graph tools.** A²Agent uses LocAgent's tools (incl. `TraverseGraph`) and adds a per-turn reward for *discovering* gold locations plus a "discovery-commit" reward; >40% of baseline failures saw the gold location but didn't commit it. Its 4B/8B models beat up-to-8× larger baselines; the localization gain carries to resolve rate (38.1% vs 31.8% for LocAgent, same repair backend) [102].
- **Process rewards for SWE agents.** SWE-TRACE (rubric process reward models), rubric-based generative reward models, Agent-RLVR guidance [103][104][39].

---

## References

Links are alphaXiv (`alphaxiv.org/abs/<id>`) or Consensus (`consensus.app/...`) as returned by the tools. "(abstract)" marks sources read only at abstract level.

1. Jimenez et al. 2023. [SWE-bench: Can Language Models Resolve Real-World GitHub Issues?](https://consensus.app/papers/details/26adfda119a0585ea6f6dad25e11f236/)
2. Zan et al. 2024. [SWE-bench-java](https://consensus.app/papers/details/00188878f1a456f499cfcf180a5d488a/) (abstract)
3. Zhang et al. 2025. [SWE-bench Goes Live!](https://consensus.app/papers/details/25ae007485d553a1aaf8bac68e19ce6f/) (abstract)
4. Deng et al. 2025. [SWE-Bench Pro](https://consensus.app/papers/details/6718bb4937655acaaff1fbd4d82be40f/) (abstract)
5. Badertdinov et al. 2025. [SWE-rebench](https://consensus.app/papers/details/84224bee0b8e5a0794a3b69ae7f3e1c1/) (abstract)
6. Aleithan et al. 2024. [SWE-Bench+: Enhanced Coding Benchmark for LLMs](https://consensus.app/papers/details/a2fde21b76f5544c9edca661a2b6e868/) (abstract)
7. Xu et al. 2026. [SWE-Bench+: Enhanced LLM Coding Benchmark](https://consensus.app/papers/details/2d2da3ec196b585a89b5d1d5ba3693b6/) (abstract)
8. Prathifkumar et al. 2025. [Does SWE-Bench-Verified Test Agent Ability or Model Memory?](https://consensus.app/papers/details/b4db7508813d5b5bb87f40a948b9ddbf/) (abstract)
9. [Coding Agents Have Converged](https://www.alphaxiv.org/abs/2609.17394) 2026 (abstract)
10. Kio. [SWE-bench-secret](https://consensus.app/papers/details/c3b54f6dbea75225bce981af625f90e0/) (abstract)
11. Huang et al. 2026. [DeepSWE: Measuring Frontier Coding Agents on Original, Long-Horizon Tasks](https://consensus.app/papers/details/a0dcab772fbd54e79116dca21157f11f/) (abstract)
12. [SWE-Explore: Benchmarking How Coding Agents Explore Repositories](https://www.alphaxiv.org/abs/2606.07297) 2026 (abstract)
13. [What Resolve Rate Hides: Trajectory Structure Diagnostics](https://www.alphaxiv.org/abs/2607.06184) 2026 (abstract)
14. [SWE-CC: Repository Policy Compliance](https://www.alphaxiv.org/abs/2610.06193) 2026 (abstract)
15. [SWE-Gate: Passing Functional Tests Is Not Enough](https://www.alphaxiv.org/abs/2609.04167) 2026 (abstract)
16. Yang et al. 2024. [SWE-agent: Agent-Computer Interfaces](https://consensus.app/papers/details/bea49f5891d7531a947c67aa8ea4b636/) (abstract)
17. Zhang et al. 2024. [AutoCodeRover](https://consensus.app/papers/details/fcabd95011765eea90d57a6b2f3af957/) (abstract)
18. Ruan et al. 2024. [SpecRover](https://consensus.app/papers/details/ccf795fd03c454028066a85b371be803/) (abstract)
19. Xia et al. 2024. [Agentless](https://consensus.app/papers/details/26a9586fd7f553fa8f75394a3bf8ff0c/) (abstract)
20. Xia et al. 2025. [Demystifying LLM-Based Software Engineering Agents (FSE)](https://consensus.app/papers/details/b8286689f24d5d35b36b865547871b8c/) (abstract)
21. Tao et al. 2024. [MAGIS](https://consensus.app/papers/details/38c840e0a60f57faa1cd3e4f404c5d61/) (abstract)
22. Zhang et al. 2026. [SGAgent](https://consensus.app/papers/details/61daed805b6e55188c41cb7e6ef89a69/) (abstract)
23. Pabba et al. 2025. [REFINE](https://consensus.app/papers/details/79e90f88129f5864b3f17ab8b29fbd29/) (abstract)
24. [SWE-Review](https://www.alphaxiv.org/abs/2607.06065) 2026 (abstract)
25. Bouzenia et al. 2025. [Understanding SWE Agents: Thought-Action-Result Trajectories](https://consensus.app/papers/details/de61aa1d8d3b5c9c819ef1957274d687/) (abstract)
26. Xu et al. 2026. [The Devil Is in the Interface (COLM 2026)](https://www.alphaxiv.org/abs/2608.11386)
27. Pan et al. 2024. [SWE-Gym](https://www.alphaxiv.org/abs/2412.21139) (abstract)
28. Jain et al. 2025. [R2E-Gym](https://www.alphaxiv.org/abs/2504.07164) (abstract)
29. SWE-smith (Yang et al. 2025), as used and described in [63].
30. [CodeMidas](https://www.alphaxiv.org/abs/2609.22068) 2026 (abstract)
31. [Scaling Agentic RL: 365,000+ Environments](https://www.alphaxiv.org/abs/2607.scaling-agentic-rl) 2026 (abstract)
32. Xie et al. 2025. [SWE-Fixer](https://consensus.app/papers/details/eda5c03fdf9c5b1696c19c0c4f32610b/) (abstract)
33. [SWE-Lego](https://www.alphaxiv.org/abs/2601.01426) 2026 (abstract)
34. [SWE-Master](https://www.alphaxiv.org/abs/2602.03411) 2026 (abstract)
35. [SWE-ZERO to SWE-HERO](https://www.alphaxiv.org/abs/2604.01496) 2026 (abstract)
36. [Open-SWE-Traces](https://www.alphaxiv.org/abs/2606.16038) 2026 (abstract)
37. Wei et al. 2025. [SWE-RL](https://consensus.app/papers/details/57c0af7ccf425bb59359e00495b71a68/) (abstract)
38. Golubev et al. 2025. [Training Long-Context, Multi-Turn SWE Agents with RL](https://consensus.app/papers/details/c4813688beb254598c74fb64bbce129e/) (abstract)
39. Da et al. 2025. [Agent-RLVR](https://consensus.app/papers/details/6b95a181e1215d22aaa879b411c9a25a/) (abstract)
40. Li et al. 2026. [BoostAPR](https://consensus.app/papers/details/4faa0c60bd5b5a9799a8d08e395905e1/) (abstract)
41. Shum et al. 2025. [SWE-RM](https://consensus.app/papers/details/a1b8f3f633d85e239f4ca57611e686ed/) (abstract)
42. Masud et al. 2026. [Reward Engineering for Software Tasks: A Survey](https://consensus.app/papers/details/addfa06bd32455528b58c71c71ae9ae2/) (abstract)
43. [SWE-MILE](https://www.alphaxiv.org/abs/2609.32631) 2026 (abstract)
44. [TRACE: Turn-level Reward Assignment](https://www.alphaxiv.org/abs/2607.13988) 2026 (abstract)
45. [Dependency-Aware Reward Shaping for Agentic RL](https://www.alphaxiv.org/abs/2610.01207) 2026 (abstract)
46. [LEGO-RL](https://www.alphaxiv.org/abs/2608.17393) 2026 (abstract)
47. [What Does Multi-Harness RL Learn?](https://www.alphaxiv.org/abs/2609.04518) 2026 (abstract)
48. Dettmers et al. 2023. [QLoRA](https://consensus.app/papers/details/d03b940acefc5c64a10b5f96c39625ed/)
49. Liu et al. 2024. [DoRA: Weight-Decomposed Low-Rank Adaptation](https://consensus.app/papers/details/9425e3ba33b3570c886d67b40a59d032/) (abstract)
50. Han et al. 2025. [MoRAgent: Mixture-of-Roles PEFT for agents](https://consensus.app/papers/details/002636daddcc5dc3a7790b281da631fe/) (abstract)
51. [FrogNano: a 4B coding agent](https://www.alphaxiv.org/abs/2609.07925) 2026 (abstract)
52. [Mingbird: local-first harness for small open models](https://www.alphaxiv.org/abs/2610.02001) 2026 (abstract)
53. [Kozuchi Agent](https://www.alphaxiv.org/abs/2608.15579) 2026 (abstract)
54. [CompactionRL](https://www.alphaxiv.org/abs/2607.05378) 2026 (abstract)
55. [AutoCompact](https://www.alphaxiv.org/abs/2610.02163) 2026 (abstract)
56. [SWE-Pruner Pro](https://www.alphaxiv.org/abs/2607.18213) 2026 (abstract)
57. [Beyond Token Savings: Context Compression in LLM Agents](https://www.alphaxiv.org/abs/2609.32961) 2026 (abstract)
58. Jang et al. 2026. [Flat Score, Amplified Failures: quantized LLM agents](https://www.alphaxiv.org/abs/2607.27275)
59. [Smaller Models, Unexpected Costs: quantization for APR](https://www.alphaxiv.org/abs/2606.27205) 2026 (abstract)
60. Ouyang et al. 2024. [RepoGraph](https://www.alphaxiv.org/abs/2410.14684) (abstract)
61. Liu et al. 2024. [CodexGraph](https://www.alphaxiv.org/abs/2408.03910) (abstract)
62. Chen et al. 2025. [LocAgent: Graph-Guided LLM Agents for Code Localization (ACL 2025)](https://www.alphaxiv.org/abs/2503.09089)
63. Sutawika et al. 2026. [CodeScout: RL recipe for code search agents](https://www.alphaxiv.org/abs/2603.17829)
64. Seddik et al. 2026. [ARISE](https://consensus.app/papers/details/299c0c8e48eb5950895c1601edaf14e8/) (abstract; also [alphaXiv 2605.03117](https://www.alphaxiv.org/abs/2605.03117))
65. Wei et al. 2026. [SemNav: Semantic Navigation for Issue Localization](https://www.alphaxiv.org/abs/2609.31176)
66. Zhang et al. 2026. [DualView: Dual-Modal Structural Reasoning for Issue Resolution](https://www.alphaxiv.org/abs/2607.01929)
67. Hu et al. 2026. [LARGER: Lexically Anchored Repository Graph Exploration](https://www.alphaxiv.org/abs/2605.16352)
68. Lin et al. 2026. [How Much Static Structure Do Code Agents Need? (CodeAnchor, ISSTA 2026)](https://www.alphaxiv.org/abs/2606.26979)
69. [Codebase-Memory: tree-sitter KGs via MCP](https://www.alphaxiv.org/abs/2603.27277) 2026 (abstract)
70. [RepoAtlas](https://www.alphaxiv.org/abs/2609.16936) 2026 (abstract)
71. [Know Before Fix](https://www.alphaxiv.org/abs/2607.11111) 2026 (abstract)
72. Wang et al. 2026. [GREPO: GNN benchmark for repository-level bug localization](https://www.alphaxiv.org/abs/2602.13921)
73. [GraphLocator](https://www.alphaxiv.org/abs/2512.22469) 2025 (abstract)
74. [SweRank+](https://www.alphaxiv.org/abs/2512.20482) 2025 (abstract)
75. Caumartin et al. 2026. [Retrieval-Oriented Code Representations in Agentic Bug Localization](https://www.alphaxiv.org/abs/2607.11046)
76. Li et al. 2019. [Context-based code representation for bug detection](https://consensus.app/papers/details/d9fa8b2e0e7153f0b270287e3041a970/) (abstract)
77. Dinella et al. 2020. [Hoppity](https://consensus.app/papers/details/d3b2d8200eb659a8a8d578312856e7f2/) (abstract)
78. Wang et al. 2020. [Graph Interval Neural Network](https://consensus.app/papers/details/d128870cee3852199634e1c72f116d24/) (abstract)
79. Nashid et al. 2023. [GLANCE: code dependencies for neural program repair](https://consensus.app/papers/details/a89f0cf9ea9b5e95908ef7fbfdf9acf0/) (abstract)
80. Li et al. 2023. [SGEPR: sequence code property graph repair](https://consensus.app/papers/details/3ffed9c3291d5704abbef7152e6c887d/) (abstract)
81. Huang et al. 2026. [CT-Repair: CPG + temporal execution graphs](https://consensus.app/papers/details/bb4b7f4cc478502386862a02ae74d1c4/) (abstract)
82. Cui et al. 2026. [StructFix](https://consensus.app/papers/details/530df3fd23165b6a9e35f60b31c891aa/) (abstract)
83. Liu et al. 2021. [GraphSearchNet](https://consensus.app/papers/details/ad9ac514540b5cbfaf963a1d0fc5ca73/) (abstract)
84. Ling et al. 2020. [Deep Graph Matching and Searching (DGMS)](https://consensus.app/papers/details/92d012a3ad925c87bd0d8e0f316e9ab9/) (abstract)
85. Zeng et al. 2021. [deGraphCS](https://consensus.app/papers/details/888965edc668542d8f570af674d83d3c/) (abstract)
86. Tao et al. 2025. [Code Graph Model (CGM)](https://www.alphaxiv.org/abs/2505.16901)
87. [Graph-aware LoRA Generation](https://www.alphaxiv.org/abs/2606.22429) 2026 (abstract)
88. [Deep Graph-Language Fusion for Structure-Aware Code Generation](https://www.alphaxiv.org/abs/2605.03689) 2026 (abstract)
89. [Bridging Code Graphs and LLMs](https://www.alphaxiv.org/abs/2512.07666) 2025 (abstract)
90. [When Graph Tokens Sink](https://www.alphaxiv.org/abs/2606.03712) 2026 (abstract)
91. Fatemi et al. 2023. [Talk like a Graph](https://consensus.app/papers/details/250a9c45a975549eb19aeafcfe9eade9/) (abstract)
92. Wang et al. 2023. [NLGraph](https://consensus.app/papers/details/b04d911a231e5f3a818369910880085d/) (abstract)
93. Guo et al. 2023. [GPT4Graph](https://consensus.app/papers/details/fc3261bff5ba576fb00bc3399e40263c/) (abstract)
94. Jin et al. 2024. [Graph Chain-of-Thought](https://consensus.app/papers/details/f7237c05e7575777af986181bd6ecd18/) (abstract)
95. [GraphCert](https://www.alphaxiv.org/abs/2609.38798) 2026 (abstract)
96. [Agentic Graph Token Reasoning](https://www.alphaxiv.org/abs/2608.00542) 2026 (abstract)
97. [Schema-Agnostic Graph Reasoning Agent](https://www.alphaxiv.org/abs/2608.15834) 2026 (abstract)
98. [GTA: Graph Theory Agent and Benchmark](https://www.alphaxiv.org/abs/2609.12265) 2026 (abstract)
99. Jin et al. 2023. [LLMs on Graphs: A Comprehensive Survey](https://consensus.app/papers/details/b7240423603a570f9d5bf76a11dbeec8/) (abstract)
100. Yu et al. 2025. [Graph2text or Graph2token](https://consensus.app/papers/details/6be0704f893c57e7a42a1035ea455e4e/) (abstract)
101. Wang et al. 2024. [Graph ML in the Era of LLMs](https://consensus.app/papers/details/893102e319da5b5f92a16accd00f4c23/) (abstract)
102. Kim et al. 2026. [A²Agent: Action-Aware RL for Code Localization Agents](https://www.alphaxiv.org/abs/2608.29831)
103. [SWE-TRACE: rubric process reward models](https://www.alphaxiv.org/abs/2604.14820) 2026 (abstract)
104. Huang et al. 2026. [Rubric-Based GRM for Reinforced Fine-Tuning SWE Agents](https://consensus.app/papers/details/cb657b491e2755c9a608e22922512e9b/) (abstract)
105. [PyCG: Practical Call Graph Generation in Python](https://www.alphaxiv.org/abs/2103.00587) 2021 (abstract)
106. [Scalable and Precise Application-Centered Call Graph Construction for Python](https://www.alphaxiv.org/abs/2305.05949) 2024 (abstract)
107. [CPyGraph: version-aware static analysis for CPython bytecode](https://www.alphaxiv.org/abs/2609.25083) 2026 (abstract)

Also consulted, not cited above: Gemma 4 Technical Report ([2607.02770](https://www.alphaxiv.org/abs/2607.02770)); RepoCoder, RepoHyper, GraphCoder, Hydra, InlineCoder, DyRetriever and RANGER (graph retrieval for repository-level code *completion*, via Consensus); Cross-Benchmark Transfer from RL on Agentic Coding ([2610.00890](https://www.alphaxiv.org/abs/2610.00890)).

---

*Consensus search notice (reproduced as required by the tool):* Upgrade to Consensus Pro to return 20 results per search instead of 10, and include more data like study design and key takeaways for every result.: https://consensus.app/pricing/
