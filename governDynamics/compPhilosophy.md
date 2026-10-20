# compPhilosophy.md

**Who this is for:** whoever (person or model) picks the research direction for our entry to the **Paper Track of the Gemma 4 Developer Agent Competition**. Read it before the literature review. It explains what the competition is for, how it is judged, and how we work.

**What this is not:** a claim about where the field stands. No literature review has been done yet. Any statement here about what works, what is open, or what is promising is a hypothesis until a published source or one of our own experiments settles it. Where the competition says something, we quote it. Where we are reasoning, we say so.

---

## 1. The problem, in the competition's own words

> "Today's best coding agents rely on massive API models running in the cloud. What if every developer could rely on an autonomous agent equally capable offline, on consumer hardware?"

> "Small local models have made rapid progress, but they still struggle with complex, multi-turn software engineering. Large codebases are hard to navigate, making reliable bug fixes and feature additions tough for models that fit on a single accelerator."

> "While The Gemma 4 Developer Agent Competition leaderboard ranks participants by bugs fixed, the Paper Track invites you to contribute research to advance the state-of-the-art in agentic software engineering."

> "Submissions can include any related research, as long as it is unpublished."

The setting: the competition "evaluates AI software engineering agents on their ability to solve real-world Python bug fixes and feature requests." There is one fixed base model, `gemma-4-31b-it-qat-w4a16-ct`, with optional LoRA adapters. Scoring is "Similar to SWE-Bench … a PASS/FAIL metric."

### 1.1 Paper Track logistics
- **Dates:** "September 22, 2026 - Start Date." "November 12, 2026 - Final Submission Deadline" (11:59 PM UTC). We joined about two weeks late. As of October 6 there are about five weeks left.
- **Required contents:** "Title and subtitle, Abstract, Introduction, Description of the research, including methods and experiments, Related works and citations."
- **Scope:** "3,000 words maximum, reporting original, unpublished research."
- **Optional:** a public notebook with code, and/or "an arXiv-ready (LaTeX/Word) PDF."
- **Non-archival:** "Submission to this competition does not prevent submission to other conferences or venues."
- **Main competition not required:** "Participation in the main competition isn't required to submit to the Paper Track."
- **Visibility:** "Winning papers will be highlighted at a Google hosted event (such as a NeurIPS expo workshop) and awarded prizes."
- **Drafts don't count:** "Any un-submitted or draft Writeups by the competition deadline will not be considered by the Judges."

---

## 2. How papers are judged

"Submissions for Paper Awards are evaluated equally across the following five criteria. Each criterion is scored on a scale from 0 (lowest) to 5 (highest), with the final score calculated as the average of all five."

| Criterion | Their wording |
|---|---|
| Novelty | "Does the work provide new insights, deepen understanding, or highlight important properties of existing methods?" |
| Quality | "How general and universal is the Submission approach beyond the competition? Does your submission translate to other similar problems? How well does your method generalize?" |
| Relevance | "Is there significance and/or potential impact of the research on the field of software engineering and agentic learning?" |
| Verifiability | "Does the paper include sufficient information to understand how an innovation works; to understand how data was obtained, analyzed, and interpreted?" |
| Clarity | "Is the paper clearly presented and written?" |

Three rules shape the strategy:
- "Your Paper rubric evaluation will not be shared with you." The judges give no feedback.
- "**In the event of a tie, the Paper that was entered first to the Competition will be the winner.**"
- The Novelty criterion does not require a new method. "New insights, deepen understanding, or highlight important properties of existing methods" all count, so a careful measurement can qualify as much as a new technique.

### 2.1 The award tracks ($35,000 total)

"The Paper Award will be awarded to three Submissions with the most points." The three awards are:

**Overall Best Paper - $15,000**
Overall best paper will be awarded to the highest score calculated as the average of all five criteria.

**Best New Resource (Dataset, library, etc) - $10,000**
A resource introduces a new dataset, tool, or software, for example. This is compared to an empirical paper which generates new knowledge/experiments. For examples of each, a new 'dataset' could leverage the provided embeddings or other sources to create a dataset that can be used for either evaluation of Gemma 4 or for fine-tuning it. A new 'tool' could be a library which uses Gemma and some other algorithm to cluster code based on underlying behavior. New 'software' could be a dashboard to explore Github repos that are similar in structure to a target repo.

**Best New Application - $10,000**
A novel application is a new use-case (e.g. using code-graphs for writing TPU kernels). Show your creative uses of code-graphs and/or the provided embeddings to create novel use-cases. This may involve creating new evaluation datasets for entirely novel tasks. It could also involve application to existing benchmarks that differ from the provided ones, and using the code-graphs or embeddings where they have not been used before.

**Our reading (not confirmed by the organizers):** Overall Best Paper rewards the empirical paper that "generates new knowledge/experiments", and our method (Section 6) produces that kind of paper. But a chain of small experiments usually needs its own measuring tools, such as a task subset, an evaluation set, or a probe of the graph tools. If one of those is new and useful by itself, it might also be a candidate for Best New Resource or Best New Application. We do not know whether one paper can be considered for more than one award. We should keep all three in view when choosing a direction, and not shape the research to fit an award.

---

## 3. The four topics: our tethers

"Participants are encouraged to submit original, unpublished research submissions on topics including, but not limited to:"

- **Tuning & Optimization:** "Parameter-efficient fine-tuning and Reinforcement Learning (RL) approaches tailored to improve SWE agent capabilities."
- **Code Comprehension:** "Novel techniques for code-graph generation, parsing, and embedding to enhance deep comprehension of complex repositories."
- **Tasks & Benchmarks:** "Development of new tasks, evaluation datasets, or resources targeting code structure, graphs, and structured code generation."
- **Graph Reasoning:** "Methods and architectures for improving Large Language Model (LLM) reasoning over large-scale graphs."

The list is "including, but not limited to," so the topics are not a fence. We treat them as **the best jumping-off points and the tethers for what we are looking for.** Whatever direction we pick should trace back to at least one of them, and ideally more than one, since they overlap. For example, a code-comprehension idea may need a benchmark to measure it (Tasks & Benchmarks) and an adapter to make the model use it (Tuning & Optimization).

**This document does not choose the topic.** The literature review chooses it, based on where the gap, or ideally gaps, turn out to be.

---

## 4. What we were given (checked against the local files)

These are the ground truths the first experiments can stand on. Anything marked *unknown* is an open question, not an assumption.

- **Tasks (`tasks.jsonl`):** "129 public development tasks": fastapi 67, rich 48, requests 13, httpx 1. Each has `problem_statement`, reference `patch`, `test_patch`, `base_commit` and `created_at` (2023-07 to 2026-06). *Checked:* `hints_text` is empty for all 129. 91 of 129 reference patches touch one source file. The median patch changes 12 lines, with a long tail up to 26 files.
- **Hidden test set:** "about 120 tasks … evenly divided between the public and private splits", "curated from a set of private repositories", built "under a similar pipeline, with identical graph generation methods, and verification standards." For the test set, "a larger frontier model can pass the case or get within a single test case of passing."
- **Graphs (`graphs/`):** "serialized NetworkX Abstract Syntax Tree (AST) call and dependency graphs". Nodes are fully qualified symbols with their full source `text`. *Checked:* all 127 distinct graphs have **only `calls` edges** (452,588 in total) and 711 to 4,597 nodes each, even though the harness lists edge filters such as `"DEFINED_IN"` and `"IMPORTS"`.
- **Embeddings (`embeddings/`):** one 256-dim `float32` vector per graph node, with keys equal to node ids. The harness says that because "the sandbox runs offline without a live neural embedding server," `search_similar_code` "resolves `query` against node keys/suffixes", so it looks up a symbol and returns similar symbols rather than searching free text. *Unknown:* which model produced the vectors and what input it saw.
- **Agent tools:** `run_command`, `submit_patch`, `get_status`, `read_file`, `edit_file`, `write_file`, `get_code_neighbors`, `search_similar_code`, `get_code_subgraph`.
- **Hard limits (`HARNESS_README.md`):** `max_model_len = 32768`; `max_loras = 8`, `max_lora_rank = 128`; submission "< 3 GiB" including adapters; 4× L4 (96 GB); "12 hours to submit patches for all tasks"; `read_file` returns at most 150 lines or 10,000 characters per call; command output is cut at 5,000 characters.
- **Grading:** the patch is applied in a fresh container, any changes to test or test-config files are reverted, `test_patch` is applied, and a task counts as resolved only if pytest exits 0 and every required test passes.
- **Sample submission:** a root coder agent plus a read-only `code_analyzer` sub-agent used as an `AgentTool`, with two placeholder rank-4 LoRAs (`main_lora`, `tool_lora`). This demonstrates "multi-agent delegation, and multi-LoRA routing."

---

## 5. Why this competition exists (our reasoning, not a quotation)

This section is the logic we use to pick a direction. It is not a claim about anyone's actual intentions.

**The sponsor.** Google puts up "$100,000" in prizes. It gets recognition for advancing the field and, less cynically, a way to push forward computing and foundational research. Its most immediate return comes from the main leaderboard: in effect, other people building better and cheaper SWE agents on its open model. That reading also explains why the task is approachable and the window short. An easier problem draws more participants, and a short window creates urgency and racing. Both make improvement more likely, and even lucky small-scale results are useful data.

**The organizer.** Our understanding is that the organizer is a PhD researcher who works on combining graphs and LLMs, and who is taking time away from their own research to run this. Unlike the sponsor, they have no large monetary stake and gain little from one other person's polished paper. What they could gain a lot from is **many people running small experiments on foundation models in their field.** Smaller models are cheaper to experiment on, and the findings may carry over to larger ones. That would explain the specific language and the strong hints on the competition page about which topics and tools to use. Whether a direction turns out better or worse, the developers on the main track are fine either way. For the paper track, steering participants toward the organizer's questions takes routine work off the organizer's plate, and it gives us direction. Both sides benefit.

**The shape of the Paper Track fits that reading:**
- The window is short and the problem simple compared with other Kaggle competitions: Python repositories, one fixed model that fits "a single accelerator", a fixed set of tools.
- The field is new, so many questions are approachable but tedious. Work like that splits well across many participants working in parallel.
- Three of the four suggested topics name graphs, code structure, or embeddings, and a graph and embedding dataset with graph tools was released with the competition.
- "Tie goes to the earlier paper" suggests the organizers expect many solid, comparable contributions rather than a few groundbreaking ones. It rewards whoever lays the groundwork and submits first.

In effect, participants are a group of researchers running parallel experiments on GPUs they pay for themselves. That is not a complaint. It tells us what kind of work the judges value, and the suggested topics tell us where they are looking.

### 5.1 What this means for us
1. **The direction has already been pointed out.** Going anywhere in the direction of the four topics, with experiments that build on each other and check ground truths, is going in the right direction.
2. **This is a race of compute over ideas.** Whoever runs more *clean, connected* experiments learns more and has more to write.
3. **Ship early, then extend.** Ties go to the earlier entry, so a complete, correct writeup submitted early has an advantage. We aim to have a submittable paper well before November 12 and improve it from there.
4. **Have findings every few days.** Each finished experiment should stand as a result on its own, so that if time runs out at any step we still have a paper. The goal is not that each result wins, but that we keep producing results.
5. **The leaderboard is a measuring instrument, not the goal.** A resolution rate means something only as part of a controlled comparison. Leaderboard tuning with no question behind it is targeted guesswork, and we do not do it.

---

## 6. How we work

### 6.1 Pick a direction, not a destination
We pick one main direction, tethered to the four topics, and let it touch the others. A good direction:
- comes out of a gap (ideally gaps) found in the literature review;
- can produce a first small result within days, not weeks;
- has an obvious next step that adds exactly one component.

### 6.2 Experiments stack
Every experiment builds on the one before it. The chain starts with the most basic ground truths: what the data actually contains, what each tool actually returns, and what the unmodified model does. Only once those are measured do we add components, one at a time. An experiment that changes two things teaches us nothing about either.

### 6.3 Ground truth before claims
Unless previous literature or another interpretable published attempt has shown something to be true, we don't assume it. We run the experiment, even when the answer seems obvious.

### 6.4 Careful language
Until the literature review is done, we write "may", "we hypothesize", "untested". After it, we cite. We never present an untested belief about the field as fact, and that includes this document.

---

## 7. The metrics we are targeting (in no particular order)

**Novelty** (found from the literature review, and the gap, ideally gaps, that you are going to look for after reading this document). This is the foundation. If we are not pursuing an entirely novel, never-been-tried question, there is no point. It should also span more than one topic, which shouldn't be an issue because of the next metric.

**Quality.** There is no point in pursuing a topic so advanced that it won't generalize. The first babystep experiment that produces a result is the one we test on the SWE benchmark and write up. If the next step generalizes, great. If not, we're set.

**"Babystep" is the most important part. This is our guiding light.**

**Verifiability.** Every experiment builds on the previous one, starting with the most common of ground truths. My goal is to win, yes, but I am a researcher at heart and this project will reflect that. Unless previous literature or another interpretable published attempt proves something to be true, we run the experiment, and every experiment adds only one new component.

**Relevance** is going to be treated as "Alicia Nash." She was John Nash's wife, and let's be honest, he was extremely lucky to find her, but at no point was he looking. His focus was on governing dynamics.

*Governing Dynamics, gentlemen.*
