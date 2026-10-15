#!/usr/bin/env python3
"""Convert public test-verified SWE agent trajectories into the swegemma harness tool format, by replay.

    python convert_trajs.py --out DIR --comp-tasks COMP/tasks.jsonl [--sources coderforge,rebench_oh,smith,kwai]
                            [--workers 6] [--cache DIR --cache-gb 8] [--limit-units N] [--max-rows N]

Sources (only rows the source marks as passing its tests):
  coderforge  togethercomputer/CoderForge-Preview, trajectories/filtered_reward1 (reward 1; SWE-rebench, SWE-smith, R2E-Gym tasks)
  rebench_oh  nebius/SWE-rebench-openhands-trajectories, resolved == 1
  smith       SWE-bench/SWE-smith-trajectories, split tool, resolved
  kwai        Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k (all successful; bash only)

Each task is checked out at the source's base state, the external-import pipeline's way (a git checkout of the base commit;
SWE-smith: mirror repo + bug patch; R2E-Gym: fix commit with its non-test files reverted). Then every trajectory is replayed:
  str_replace_editor view   -> read_file (in 150-line / 10k-char chunks, so the model still sees what the source saw);
                               the source's `cat -n` text is compared with the checkout, and any differing line rejects the row
  str_replace / insert      -> edit_file, applied with the harness's own swegemma.edit.apply_replacement (same diff and errors)
  create / undo_edit        -> write_file
  view of a directory       -> run_command `find <dir> -maxdepth 2 ...`, output recomputed on the checkout
  bash / execute_bash       -> run_command; the output is the source's (it ran in the source's container), with the harness's
                               JSON shape, 5000-char limit and exit code (mini-swe-agent / OpenHands report it; SWE-agent does
                               not, so it is inferred and the row is labelled exit_codes_inferred)
  think / task_tracker      -> folded into the next call's reasoning
  finish / submit           -> submit_patch (patch size from the replayed checkout) + the final text answer
Rows are rejected (rejects.jsonl.gz, with the reason) for: an editor call whose outcome differs from the source's (stale file,
edit that only one side could apply), editor paths outside the repository, interactive input (is_input) or soft-timeout
continuations the harness cannot express, missing task metadata or checkout failure. Tasks whose pull request is a
competition task are dropped (no dev/holdout leakage).

Output (Kaggle-ready, self-contained): train/<source>/<unit>.jsonl.gz rows in build_rft_data.py's format plus labels:
  {"instance_id", "repo", "source", "source_dataset", "source_traj_id", "source_model", "scaffold", "source_resolved": true,
   "base": {checkout recipe}, "labels": {...}, "n_turns", "n_chars", "messages": [system, user, assistant/tool ..., final]}
tasks.jsonl (one per task: repo, checkout recipe, problem statement), rejects, manifest.json (counts per source and reason).
Units are written atomically, so a rerun resumes where it stopped.
"""
import argparse, fcntl, gzip, json, os, re, shutil, subprocess, sys, tempfile, time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

VERSION = "4"                                               # outputs of another version are discarded on start
MAX_LINES, MAX_CHARS, MAX_OUT = 150, 10000, 5000           # harness: max_file_lines, max_file_chars, max_stdout_chars
SOURCES = {
    "coderforge": ("togethercomputer/CoderForge-Preview", "trajectories/filtered_reward1-*.parquet", "openhands", None),
    "rebench_oh": ("nebius/SWE-rebench-openhands-trajectories", "trajectories.parquet", "openhands", "Qwen3-Coder-480B-A35B-Instruct"),
    "smith": ("SWE-bench/SWE-smith-trajectories", "data/tool-*.parquet", "swe-agent", None),
    "kwai": ("Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k", "data/train-*.parquet", "mini-swe-agent", None),
}
CF_MODEL = "Qwen3-Coder-480B-A35B-Instruct"     # CoderForge-Preview's teacher (Together AI blog, Feb 2026)
R2E_REPOS = {"aiohttp": "aio-libs/aiohttp", "pandas": "pandas-dev/pandas", "numpy": "numpy/numpy", "pillow": "python-pillow/Pillow",
             "tornado": "tornadoweb/tornado", "scrapy": "scrapy/scrapy", "coveragepy": "nedbat/coveragepy", "datalad": "datalad/datalad",
             "pyramid": "Pylons/pyramid", "orange3": "biolab/orange3", "sympy": "sympy/sympy", "bokeh": "bokeh/bokeh"}
COMP_REPOS = {"psf/requests": "requests", "tiangolo/fastapi": "fastapi", "fastapi/fastapi": "fastapi", "Textualize/rich": "rich",
              "encode/httpx": "httpx"}
TEST_PATH = re.compile(r"(^|/)(tests?|testing)/|(^|/)test_[^/]*$|_tests?\.py$|(^|/)conftest\.py$")
os.environ["GIT_TERMINAL_PROMPT"] = "0"                     # deleted / private repos fail instead of asking for a login
GIT = ["git", "-c", "user.name=x", "-c", "user.email=x@x", "-c", "advice.detachedHead=false"]

# The deployed prompt (swegemma build_agent_prompt with the deploy config, sandbox testing on; no graph section because
# these tasks have no code graph). train_lora.py --deploy-budget / --system-prompt rewrite budget and system prompt anyway.
PROMPT_TAIL = """## Task Budget (Session terminates when any budget is exhausted)
- Time allowance: 30.0 minutes
- Tool calls allowance: 50 calls
- Max loop iterations: 100 turns

## Execution Environment Rules
- Single command timeout: 300 seconds (commands exceeding this fail without ending the session)
- Command output limit: 5000 characters
- File view limit: 150 lines per read_file call
- File character limit: 10000 characters per read_file call
- Environment is offline (no network/PyPI access). All repository and test dependencies are ALREADY pre-installed. Do NOT attempt to run pip install or download packages.

## Instructions:
0. All source code is under `/workspace`. Do NOT search outside `/workspace` (e.g. `/usr/`, `/wheels/`, system site-packages). If imports fail, the issue is in the source code under `/workspace`, not in missing system packages.
1. Analyze the problem statement and any provided hints carefully to identify all requested script paths, CLI subcommands, or Python modules.
2. Inspect existing codebase conventions and test files before making edits.
3. Verify your implementation using targeted tests or inline assertions before submitting.
4. Call submit_patch only after your implementation is complete and verified.
5. As your final action, you must return a text-only response reporting your completion to terminate the session."""
TREE_CMD = ("find . -mindepth 1 -maxdepth 3 -not -path \"./.*\" -not -name \".*\" -not -name \"*.pyc\" -not -name \"__pycache__\" "
            "-not -path \"*/__pycache__/*\" -not -path \"./*.egg-info*\" {prune} | awk -F/ '{{print NF, $0}}' | sort -n -k1,1 -k2 "
            "| head -200 | cut -d' ' -f2- | sort")


class Reject(Exception):
    pass


def jdump(d):
    return json.dumps(d)              # the harness's tool message is json.dumps of the tool's dict (ASCII-escaped)


def ok(**kw):
    return jdump({"status": "ok", **kw})


def err(etype, msg, details=None):
    d = {"status": "error", "error_type": etype, "error_message": msg}
    if details is not None:
        d["details"] = details
    return jdump(d)


def as_args(a):
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except Exception:
            return {}
    return a if isinstance(a, dict) else {}


def text_of(c):
    if isinstance(c, list):
        return "".join(x.get("text", "") for x in c if isinstance(x, dict))
    return c or ""


PS_TAGS = re.compile(r"<(issue_description|pr_description|github_issue|problem_statement)>\s*(.*?)\s*</\1>", re.S)


def problem_from(user_text):
    m = PS_TAGS.search(user_text)
    if not m:
        return None
    ps = m.group(2)
    return ps.replace("Consider the following PR description:\n", "", 1).strip()


# ---------------------------------------------------------------- source rows -> normalized episodes
def normalize(src, row):
    """-> episode dict, or raises Reject. steps: {thought, name, args, obs, exit, is_input}."""
    scaffold = SOURCES[src][2]
    if src == "kwai":
        return normalize_mini(row)
    msgs = row.get("messages") if src != "rebench_oh" else row.get("trajectory")
    msgs = json.loads(msgs) if isinstance(msgs, str) else list(msgs or [])
    users = [m for m in msgs if m["role"] == "user"]
    if not users:
        raise Reject("no user message")
    u0 = text_of(users[0]["content"])
    root = (re.search(r"<uploaded_files>\s*(\S+?)/?\s*</uploaded_files>", u0) or re.search(r"(/testbed)", u0))
    ep = {"src": src, "scaffold": scaffold, "root": root.group(1) if root else "/testbed", "problem": problem_from(u0),
          "steps": [], "final": None, "src_patch": row.get("model_patch") or ""}   # SWE-smith-trajectories' patch column is not this row's
    if src == "coderforge":
        ep["traj_id"], ep["model"], ep["image"] = row["trajectory_id"], CF_MODEL, row.get("image") or ""
    elif src == "rebench_oh":
        ep["traj_id"], ep["model"], ep["instance"] = row["trajectory_id"], SOURCES[src][3], row["instance_id"]
    else:
        ep["traj_id"], ep["model"], ep["instance"] = row["traj_id"], row.get("model"), row["instance_id"]
    results = {}
    for m in msgs:
        if m["role"] == "tool":
            ids = m.get("tool_call_ids") or [m.get("tool_call_id")]
            for i in ids:
                results[i] = text_of(m.get("content"))
    pending = ""
    for m in msgs:
        if m["role"] != "assistant":
            continue
        content = text_of(m.get("content")) or m.get("thought") or ""
        tcs = m.get("tool_calls") or []
        if not tcs:
            pending = (pending + "\n\n" + content).strip()
            continue
        for k, tc in enumerate(tcs):
            f = tc["function"]
            ep["steps"].append({"thought": ((pending + "\n\n" + content).strip() if k == 0 else ""), "name": f["name"],
                                "args": as_args(f.get("arguments")), "obs": results.get(tc.get("id"), None)})
            pending = ""
    if pending:
        ep["final_text"] = pending
    return ep


MINI_BLOCK = re.compile(r"```bash\s*\n(.*?)\n```", re.S)


def normalize_mini(row):
    msgs = row["messages"]
    msgs = json.loads(msgs) if isinstance(msgs, str) else list(msgs)
    u0 = text_of(msgs[1]["content"])
    ep = {"src": "kwai", "scaffold": "mini-swe-agent", "root": "/testbed", "problem": problem_from(u0), "steps": [],
          "final": None, "src_patch": "", "traj_id": row["instance_id"] + "#kwai", "model": None, "instance": row["instance_id"]}
    for i, m in enumerate(msgs[2:], start=2):
        if m["role"] != "assistant":
            continue
        t = text_of(m["content"])
        blocks = MINI_BLOCK.findall(t)
        if len(blocks) != 1:
            raise Reject("mini-swe-agent turn without exactly one bash block")
        thought = re.sub(r"^THOUGHT:\s*", "", t[: t.find("```bash")].strip())
        cmd = blocks[0].strip()
        nxt = text_of(msgs[i + 1]["content"]) if i + 1 < len(msgs) else ""
        if re.search(r"echo (COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT|MINI_SWE_AGENT_FINAL_OUTPUT)", cmd):
            pre = re.split(r"(&&|;)?\s*echo (COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT|MINI_SWE_AGENT_FINAL_OUTPUT)", cmd)[0].strip()
            if pre:                                 # e.g. `rm scratch.py && echo ...`: the cleanup runs, then the submission
                ep["steps"].append({"thought": thought, "name": "bash", "args": {"command": pre}, "obs": "", "exit": 0})
                thought = ""
            ep["steps"].append({"thought": thought, "name": "submit", "args": {}, "obs": None})
            ep["final_text"] = thought
            break
        rc = re.search(r"<returncode>(-?\d+)</returncode>", nxt)
        out = re.search(r"<output>\n?(.*?)</output>", nxt, re.S)
        if rc is None:
            raise Reject("mini-swe-agent observation without returncode (format error turn)")
        ep["steps"].append({"thought": thought, "name": "bash", "args": {"command": cmd},
                            "obs": out.group(1) if out else nxt, "exit": int(rc.group(1))})
    return ep


# ---------------------------------------------------------------- task metadata and checkout recipes
def load_meta(cache_dir):
    """instance tables for SWE-rebench (V1 filtered+test, V2 python) and SWE-smith; cached as json."""
    p = Path(cache_dir) / "task_meta.json"
    if p.exists():
        return json.loads(p.read_text())
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    reb, smith = {}, {}
    for ds, cols in (("nebius/SWE-rebench", ["instance_id", "repo", "base_commit", "problem_statement", "hints_text", "created_at", "test_patch"]),
                     ("nebius/SWE-rebench-V2", ["instance_id", "repo", "base_commit", "problem_statement", "created_at", "language", "test_patch"])):
        for f in sorted(fs.glob(f"datasets/{ds}/data/*.parquet")):
            for r in pq.read_table(fs.open(f, block_size=1 << 22), columns=cols).to_pylist():
                if r.get("language", "python").lower() == "python" and r["instance_id"] not in reb:
                    reb[r["instance_id"]] = {k: r.get(k) or "" for k in ("repo", "base_commit", "problem_statement", "hints_text", "created_at", "test_patch")}
    for f in sorted(fs.glob("datasets/SWE-bench/SWE-smith/data/*.parquet")):
        for r in pq.read_table(fs.open(f, block_size=1 << 22), columns=["instance_id", "repo", "patch", "problem_statement"]).to_pylist():
            smith[r["instance_id"]] = {"repo": r["repo"], "patch": r["patch"], "problem_statement": r["problem_statement"]}
    meta = {"rebench": reb, "smith": smith}
    p.write_text(json.dumps(meta))
    return meta


class Resolver:
    def __init__(self, meta, comp_ids):
        self.reb, self.smith = meta["rebench"], meta["smith"]
        self.reb_lc = {k.lower(): k for k in self.reb}
        self.smith_us = {k.replace(".", "_"): k for k in self.smith}
        self.comp = comp_ids                       # {(repo_short, pr_number)}

    def rebench(self, iid):
        k = iid if iid in self.reb else self.reb_lc.get(iid.lower())
        if not k:
            return None
        r = self.reb[k]
        short = COMP_REPOS.get(r["repo"])
        if short and (short, re.search(r"(\d+)$", k).group(1)) in self.comp:
            raise Reject("competition pull request")
        return {"instance_id": k, "repo": r["repo"], "problem": r["problem_statement"], "hints": r.get("hints_text") or "",
                "test_files": sorted(set(re.findall(r"(?m)^diff --git a/(\S+) b/", r.get("test_patch") or ""))),
                # SWE-rebench containers carry the PR's test patch (the source views show the new tests)
                "base": {"kind": "commit", "url": f"https://github.com/{r['repo']}.git", "commit": r["base_commit"], "test_patch": r.get("test_patch") or ""}}

    def smith_task(self, iid):
        k = iid if iid in self.smith else self.smith_us.get(iid)
        if not k:
            return None
        r = self.smith[k]
        mirror = r["repo"]                          # swesmith/<owner>__<repo>.<commit8>
        owner_repo = mirror.split("/", 1)[1].rsplit(".", 1)[0].replace("__", "/", 1)
        return {"instance_id": k, "repo": owner_repo, "problem": r["problem_statement"], "hints": "",
                "base": {"kind": "smith", "url": f"https://github.com/{mirror}.git", "commit": "HEAD", "bug_patch": r["patch"]}}

    def task_for(self, ep):
        src = ep["src"]
        if src == "rebench_oh":                    # nebius's containers: the bare base commit (no test patch) + issue.md
            t = self.rebench(ep["instance"])
            if t:
                t["base"] = {k: v for k, v in t["base"].items() if k != "test_patch"}
                t["instance_id"] += "@base"
        elif src in ("smith", "kwai"):
            t = self.smith_task(ep["instance"])
        else:
            img, tid = ep["image"], re.sub(r"_run\d+$", "", ep["traj_id"])
            if "/sweb.eval." in img:
                t = self.rebench(tid)
                if t is None:
                    m = re.search(r"sweb\.eval\.x86_64\.(.+?)_1776_(.+)$", img)
                    t = self.rebench(f"{m.group(1)}__{m.group(2)}") if m else None
            elif "/swesmith." in img:
                t = self.smith_task(tid)
            else:
                m = re.search(r"/([\w.-]+?)_final:([0-9a-f]{40})$", img)
                if not m or m.group(1) not in R2E_REPOS:
                    raise Reject(f"R2E repo unknown ({img})")
                repo = R2E_REPOS[m.group(1)]
                t = {"instance_id": f"r2e__{m.group(1)}__{m.group(2)[:12]}", "repo": repo, "problem": None, "hints": "",
                     "base": {"kind": "r2e", "url": f"https://github.com/{repo}.git", "commit": m.group(2)}}
        if t is None:
            raise Reject("task metadata not found")
        return t


# ---------------------------------------------------------------- checkouts (partial clones, one lock per repo)
def sh(args, cwd=None, check=True, timeout=1800):
    r = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=timeout, errors="replace")
    if check and r.returncode:
        raise RuntimeError(f"{' '.join(map(str, args[:6]))}: {r.stderr.strip()[-300:]}")
    return r


class Repo:
    def __init__(self, cache, url):
        self.dir = Path(cache) / re.sub(r"[^\w.-]", "_", url.split("github.com/")[1].removesuffix(".git"))
        self.url, self.lock = url, Path(str(self.dir) + ".lock")

    def __enter__(self):
        self.dir.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.lock, "w")
        fcntl.flock(self.fh, fcntl.LOCK_EX)
        if not self.dir.exists():
            tmp = Path(str(self.dir) + ".tmp")
            shutil.rmtree(tmp, ignore_errors=True)
            sh(GIT + ["clone", "--bare", "--quiet", "--filter=blob:none", self.url, str(tmp)], timeout=3600)
            tmp.rename(self.dir)
        (self.dir / "LAST_USED").write_text(str(time.time()))
        return self

    def __exit__(self, *a):
        fcntl.flock(self.fh, fcntl.LOCK_UN)
        self.fh.close()

    def has(self, commit):
        return sh(GIT + ["-C", str(self.dir), "cat-file", "-e", f"{commit}^{{commit}}"], check=False).returncode == 0

    def checkout(self, commit, dest):
        if commit != "HEAD" and not self.has(commit):
            sh(GIT + ["-C", str(self.dir), "fetch", "--quiet", "--filter=blob:none", "origin", commit], timeout=3600)
        sh(GIT + ["-C", str(self.dir), "worktree", "add", "--quiet", "--detach", "-f", str(dest), commit], timeout=3600)


def prepare(task, cache, dest):
    """Check out the task's base state into dest and commit it as the baseline; returns the workspace tree string."""
    b = task["base"]
    with Repo(cache, b["url"]) as repo:
        repo.checkout(b["commit"], dest)
        if b["kind"] == "r2e":                     # R2E-Gym: fix commit, non-test files reverted to the parent
            changed = sh(GIT + ["diff", "--name-only", "--no-renames", f"{b['commit']}^", b["commit"]], cwd=dest).stdout.split()
            for f in changed:
                if TEST_PATH.search(f):
                    continue
                if sh(GIT + ["cat-file", "-e", f"{b['commit']}^:{f}"], cwd=dest, check=False).returncode == 0:
                    sh(GIT + ["checkout", f"{b['commit']}^", "--", f], cwd=dest)
                elif (Path(dest) / f).exists():
                    (Path(dest) / f).unlink()
    for key in ("bug_patch", "test_patch"):          # SWE-smith: mirror + bug patch; SWE-rebench: base commit + test patch
        if b.get(key):
            p = Path(dest) / ".x.patch"
            p.write_text(b[key])
            sh(GIT + ["apply", "--whitespace=nowarn", ".x.patch"], cwd=dest)
            p.unlink()
    sh(GIT + ["add", "-A"], cwd=dest)
    sh(GIT + ["commit", "-q", "--allow-empty", "--no-verify", "-m", "baseline"], cwd=dest)
    prune = " ".join(f'-not -path "./{d}/*"' for d in ("node_modules", "venv", ".venv", "build", "dist"))
    return sh(["bash", "-c", TREE_CMD.format(prune=prune)], cwd=dest, check=False).stdout.strip()


# ---------------------------------------------------------------- replay
def apply_replacement(content, old, new, filepath):
    from swegemma.edit import apply_replacement as ar      # the harness's own matcher (exact -> flexible -> regex)
    return ar(content, old, new, filepath=filepath)


def numbered(obs):
    """{line_no: text} from a `cat -n` observation (OpenHands / SWE-agent view)."""
    out = {}
    for l in obs.replace("\r\n", "\n").split("\n"):
        m = re.match(r"^\s*(\d+)\t(.*)$", l)
        if m and "<response clipped>" not in m.group(2):
            out[int(m.group(1))] = m.group(2)
    return out


def read_chunks(lines, start, end):
    """[(s, e)] read_file ranges covering start..end within the harness's 150-line / 10k-char view limits."""
    out, s = [], start
    while s <= end:
        e = min(end, s + MAX_LINES - 1)
        while e > s and len("\n".join(lines[s - 1:e])) > MAX_CHARS:
            e = s + (e - s) // 2 if len("\n".join(lines[s - 1:s - 1 + (e - s) // 2 + 1])) > MAX_CHARS else e - 1
        out.append((s, e))
        s = e + 1
    return out


def read_obs(rel, lines, s, e, full):
    snip = "\n".join(lines[s - 1:e])
    trunc = len(snip) > MAX_CHARS
    return ok(filepath=rel, content=snip[:MAX_CHARS], start_line=s, end_line=e, total_lines=len(lines), is_truncated=trunc)


OH_FOOTER = re.compile(r"\n?\[(The command completed with exit code (-?\d+)\.|Current working directory: .*|Python interpreter: .*|"
                       r"Command finished with exit code (-?\d+)|Below is the output of the previous command\.)\]", re.M)


def bash_result(step, scaffold):
    """(stdout_text, exit_code, inferred) from the source observation."""
    o = step["obs"] or ""
    if scaffold == "mini-swe-agent":
        return o, step["exit"], False
    if scaffold == "openhands":
        codes = [int(a or b) for _, a, b in OH_FOOTER.findall(o) if (a or b)]
        if "has no new output after" in o or (codes and codes[-1] == -1):
            raise Reject("soft-timeout continuation (not expressible in the harness)")
        return OH_FOOTER.sub("", o).strip("\n"), (codes[-1] if codes else 0), not codes
    o = re.sub(r"^OBSERVATION:\n?", "", o)
    if o.strip() == "Your command ran successfully and did not produce any output.":
        return "", 0, True
    bad = re.search(r"Traceback \(most recent call last\)|command not found|No such file or directory|^\w*Error: ", o, re.M)
    return o, (1 if bad else 0), True


SOFT_NOTE = re.compile(r"\n?\[(The command has no new output after .*?|Your command .*? is NOT executed.*?|.*?CTRL\+C was sent\.)\]", re.S)
KEYS = ("", "C-c", "C-d", "C-z")


def merge_soft_timeouts(steps):
    """OpenHands returns after 10 s without output (exit -1); the agent then waits ('') or interrupts (C-c). The harness
    runs a command to completion (300 s), so such a sequence is one run_command: waits that end in completion -> the
    concatenated output with the final exit code; an interrupt -> the harness's TimeoutExceeded error. Commands OpenHands
    refused while the previous one ran ("is NOT executed") had no effect and are dropped."""
    out, i = [], 0
    while i < len(steps):
        st = steps[i]
        o = st.get("obs") or ""
        hung = st["name"] == "execute_bash" and ("no new output after" in o or re.search(r"exit code -1\b", o))
        if not hung:
            if st["name"] == "execute_bash" and "is NOT executed" in o:
                i += 1
                continue
            out.append(st)
            i += 1
            continue
        parts, interrupted, code, j = [SOFT_NOTE.sub("", o)], False, None, i + 1
        while j < len(steps) and steps[j]["name"] == "execute_bash":
            a = steps[j]["args"]
            c = str(a.get("command") or "").strip()
            cont = a.get("is_input") in (True, "true", "True") or c in KEYS
            oj = steps[j].get("obs") or ""
            if not cont:
                if "is NOT executed" in oj:            # refused while the hung command ran: no effect
                    j += 1
                    continue
                break
            if c not in ("",) and c not in KEYS:
                raise Reject("interactive input to a running command")
            interrupted |= c in ("C-c", "C-d", "C-z")
            parts.append(SOFT_NOTE.sub("", oj))
            j += 1
            m = re.findall(r"exit code (-?\d+)", oj)
            if m and m[-1] != "-1":
                code = int(m[-1])
                break
        merged = dict(st, obs=OH_FOOTER.sub("", "\n".join(parts)).strip("\n"))
        if interrupted or code is None:
            merged["timeout"] = True
        else:
            merged["obs"] += f"\n[The command completed with exit code {code}.]"
        merged["merged_soft_timeout"] = True
        out.append(merged)
        i = j
    return out


GIT_CMD = re.compile(r"\bgit\b[^|;&]*")
GIT_OTHER_REFS = re.compile(r"--all\b|--branches|--remotes|--tags\b|\borigin/|FETCH_HEAD|\breflog\b|\bfsck\b|--glob")
MAX_EDITOR_ERRORS = 2                                     # SWE-HERO prunes trajectories with >= 3 str_replace_editor errors


class Replay:
    def __init__(self, ep, task, ws, tree):
        self.ep, self.task, self.ws, self.tree = ep, task, Path(ws), tree
        self.root = ep["root"].rstrip("/")
        self.msgs, self.k, self.hist = [], 0, defaultdict(list)
        self.labels = Counter()

    # paths and text
    def rel(self, p):
        p = str(p or "").rstrip("/")
        parent = self.root.rsplit("/", 1)[0] or "/"
        if p in (self.root, parent, "", "."):
            return "."
        if p.startswith(self.root + "/"):
            return p[len(self.root) + 1:]
        if p.startswith("/"):
            raise Reject("editor path outside the repository")
        return p

    def fix(self, s):
        """The source's repository root -> /workspace (not inside longer paths such as /opt/conda/envs/testbed)."""
        s = str(s or "")
        return re.sub(r"(?<![\w.-])" + re.escape(self.root) + r"(?![\w.-])", "/workspace", s) if self.root != "/workspace" else s

    def call(self, thought, name, args, obs):
        self.k += 1
        cid = f"call_{self.k}"
        self.msgs.append({"role": "assistant", "content": "", "reasoning_content": self.fix(thought),
                          "tool_calls": [{"id": cid, "type": "function", "function": {"name": name, "arguments": args}}]})
        self.msgs.append({"role": "tool", "tool_call_id": cid, "name": name, "content": obs})

    def file(self, rel):
        p = (self.ws / rel).resolve()
        if not str(p).startswith(str(self.ws.resolve())):
            raise Reject("editor path outside the repository")
        return p

    def run(self):
        ep, sc = self.ep, self.ep["scaffold"]
        thought, submitted = "", False
        steps = merge_soft_timeouts(ep["steps"]) if sc == "openhands" else ep["steps"]
        for st in steps:
            name, a = st["name"], st["args"]
            th = (thought + "\n\n" + (st["thought"] or "")).strip()
            thought = ""
            if a.get("is_input") in (True, "true", "True"):
                raise Reject("interactive input to a running command")
            if name == "think":
                thought = (th + "\n\n" + str(a.get("thought") or a.get("content") or "")).strip()
                continue
            if name == "task_tracker":
                thought = th
                continue
            if name in ("finish", "submit"):
                self.submit(th)
                submitted = True
                if a.get("message"):
                    ep["final_text"] = str(a["message"])
                continue
            if name in ("execute_bash", "bash"):
                cmd = str(a.get("command") or "")
                if not cmd.strip() or cmd.strip() in ("C-c", "C-d", "C-z"):
                    raise Reject("interactive input to a running command")
                self.git_hack(cmd)
                self.labels["soft_timeouts_merged"] += bool(st.get("merged_soft_timeout"))
                if st.get("timeout"):
                    obs = err("TimeoutExceeded", "Command timed out after 300 seconds")
                else:
                    out, code, inferred = bash_result(st, sc)
                    self.labels["exit_codes_inferred"] |= inferred
                    out = self.fix(out)[:MAX_OUT]
                    obs = ok(stdout=out, stderr="", exit_code=0) if code == 0 else \
                        err("CommandError", out, {"stdout": out, "stderr": "", "exit_code": code})
                self.call(th, "run_command", {"command": self.fix(cmd)}, obs)
                self.mirror_rm(cmd)
                if re.search(r"\b(pip|pip3|conda|apt-get|apt) (install|download)\b|\b(curl|wget) ", cmd):
                    self.labels["network_or_install_cmds"] += 1      # offline at deploy: filter on this label if wanted
                if re.search(r"(sed -i|>\s*[\w./-]+\.py|\btee\b|\bmv\b|\bpatch\b|git (apply|checkout|stash|reset))", cmd):
                    self.labels["bash_may_write"] = 1
                continue
            if name == "str_replace_editor":
                self.editor(th, a, st["obs"] or "")
                continue
            raise Reject(f"unknown tool {name}")
        if not submitted:
            if sc == "openhands":
                raise Reject("trajectory does not end with finish")
            self.submit("")
        final = ep.get("final_text") or ""
        if not final.strip():
            final = next((m["reasoning_content"] for m in reversed(self.msgs) if m["role"] == "assistant" and m["reasoning_content"].strip()), "")
            self.labels["final_from_reasoning"] = 1
        self.msgs.append({"role": "assistant", "content": self.fix(final)})
        changed = norm_diff(self.final_diff)               # modified pre-existing files of the submitted patch
        tests = set(self.task.get("test_files") or [])
        if any(f in tests or TEST_PATH.search(f) for f in changed):
            raise Reject("modifies test files (shortcut)")   # SWE-Lego / SWE-HERO: cheating-by-tests filter
        new = re.findall(r"(?m)^diff --git a/(\S+) b/\S+\nnew file mode", self.final_diff)
        if not [f for f in changed if not TEST_PATH.search(f)] and not [f for f in new if "/" in f and not TEST_PATH.search(f)]:
            raise Reject("no source change")               # SWE-HERO: null code changes (root-level new files = scratch)
        return self.msgs

    def git_hack(self, cmd):
        """SWE-Lego / SWE-HERO 'git hacking': reading commits after the base (the fix) or other refs. Past history is
        legitimate (the competition snapshots ship .git), so a commit named in a git command must be an ancestor of HEAD."""
        for g in GIT_CMD.findall(cmd):
            if GIT_OTHER_REFS.search(g):
                raise Reject("git hacking: other refs / all branches")
            for sha in re.findall(r"\b[0-9a-f]{7,40}\b", g):
                if sh(GIT + ["rev-parse", "--verify", "-q", f"{sha}^{{commit}}"], cwd=self.ws, check=False).returncode == 0 and \
                        sh(GIT + ["merge-base", "--is-ancestor", sha, "HEAD"], cwd=self.ws, check=False).returncode != 0:
                    raise Reject("git hacking: commit after the base")

    def submit(self, thought):
        sh(GIT + ["add", "-N", "."], cwd=self.ws, check=False)
        diff = sh(GIT + ["diff", "--binary", "HEAD"], cwd=self.ws, check=False).stdout
        self.call(thought, "submit_patch", {}, ok(patch_size=len(diff), files_changed=diff.count("diff --git")))
        self.final_diff = diff

    def mirror_rm(self, cmd):
        """Mirror plain `rm` of files inside the repo, so the submitted diff drops deleted scratch scripts as the source did."""
        for part in re.split(r"&&|;", cmd):
            w = part.split()
            if w and w[0] == "rm":
                for f in w[1:]:
                    if f.startswith("-") or any(ch in f for ch in "*?$`"):
                        continue
                    try:
                        p = self.file(self.rel(f))
                    except Reject:
                        continue
                    if p.is_file():
                        p.unlink()

    def editor(self, th, a, src_obs):
        cmd, rel = a.get("command"), self.rel(a.get("path"))
        if rel == "issue.md" and self.ep["src"] == "rebench_oh":   # nebius's copy of the problem statement: not in the harness
            self.labels["dropped_issue_md_calls"] += 1
            return
        p = self.file(rel)
        head = re.sub(r"^OBSERVATION:\s*", "", src_obs.lstrip())[:400]
        self.labels["editor_errors"] += not re.search(r"^Here's the|^File created successfully|has been edited|undone successfully", head)
        if self.labels["editor_errors"] > MAX_EDITOR_ERRORS:
            raise Reject("3 or more editor errors")
        src_failed = not re.search({"view": r"^Here's the (result of running `cat -n`|files and directories)",
                                    "create": r"^File created successfully", "undo_edit": r"undone successfully|^Last edit to .* undone",
                                    }.get(cmd, r"has been edited|^The file .* has been edited"), head)
        if cmd == "view":
            if p.is_dir() or rel == ".":
                if src_failed:
                    raise Reject("directory view failed in the source")
                c = f"find {rel} -maxdepth 2 -not -path '*/.*' | sort"
                out = sh(["bash", "-c", c], cwd=self.ws, check=False).stdout[:MAX_OUT]
                self.call(th, "run_command", {"command": c}, ok(stdout=out, stderr="", exit_code=0))
                return
            vr = a.get("view_range")
            if isinstance(vr, str):
                vr = json.loads(vr)
            full = (not vr or (int(vr[0]) <= 1 and vr[1] in (-1, None))) and "<response clipped>" not in src_obs
            if not p.is_file():
                if src_failed:
                    self.call(th, "read_file", {"filepath": rel}, err("FileReadError", f"File '{rel}' not found in workspace."))
                    return
                if not full:
                    raise Reject("viewed file missing in the checkout")
                self.resync(p, rel, src_obs)                     # made by a shell command (cat > f, cp, ...): rebuild from the view
                self.labels["files_rebuilt_from_view"] += 1
            if src_failed:                  # e.g. a view_range past the end: the harness's read_file would clamp it
                self.labels["dropped_failed_noop_calls"] += 1
                return
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
            if not self.matches(lines, numbered(src_obs)):
                if not full:
                    raise Reject("stale file: source view differs from the replayed checkout")
                self.resync(p, rel, src_obs)                     # changed by a shell command (sed -i, ...): take the source's state
                self.labels["files_resynced_from_view"] += 1
                lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
            s, e = (vr[0], (len(lines) if vr[1] in (-1, None) else vr[1])) if vr else (1, len(lines))
            s, e = max(1, int(s)), min(len(lines), int(e))
            if not lines or s > e:
                self.call(th, "read_file", {"filepath": rel}, ok(filepath=rel, content="", start_line=1, end_line=0,
                                                                    total_lines=0, is_truncated=False))
                return
            chunks = read_chunks(lines, s, e)
            self.labels["chunked_views"] += len(chunks) > 1
            for i, (cs, ce) in enumerate(chunks):
                args = {"filepath": rel} if (cs, ce) == (1, len(lines)) else {"filepath": rel, "start_line": cs, "end_line": ce}
                self.call(th if i == 0 else "", "read_file", args, read_obs(rel, lines, cs, ce, False))
            return
        if cmd == "create":
            if src_failed:
                self.labels["dropped_failed_noop_calls"] += 1
                return
            txt = str(a.get("file_text") or "")
            self.hist[rel].append(p.read_text(errors="replace") if p.exists() else None)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(txt)
            self.call(th, "write_file", {"filepath": rel, "content": self.fix(txt)}, ok(filepath=rel, size=len(self.fix(txt))))
            return
        if cmd in ("str_replace", "insert"):
            if not p.is_file():
                if src_failed:
                    self.labels["dropped_failed_noop_calls"] += 1
                    return
                raise Reject("edited file missing in the checkout")
            content = p.read_text(encoding="utf-8", errors="replace")
            if cmd == "str_replace":
                old, new = str(a.get("old_str") or ""), str(a.get("new_str") or "")
            else:
                old, new = self.insert_anchor(content, int(a.get("insert_line") or 0), str(a.get("new_str") or ""))
            try:
                res = apply_replacement(content, old, new, rel)
                ours_ok = True
            except ValueError as e:
                res, ours_ok = str(e), False
            if ours_ok != (not src_failed):
                if src_failed:              # the source's edit failed, ours would apply: drop the no-op call
                    self.labels["dropped_failed_noop_calls"] += 1
                    return
                raise Reject("edit applies in the source but not in the replayed checkout")
            args = {"filepath": rel, "old_string": self.fix(old), "new_string": self.fix(new)}
            if not ours_ok:
                self.call(th, "edit_file", args, err("FileEditError", res))
                return
            self.hist[rel].append(content)
            p.write_text(res.new_content)
            diff = res.diff
            self.call(th, "edit_file", args, ok(filepath=rel, occurrences=res.occurrences, strategy=res.strategy,
                                                diff=diff[:MAX_OUT], is_truncated=len(diff) > MAX_OUT))
            return
        if cmd == "undo_edit":
            if src_failed or not self.hist[rel]:
                raise Reject("undo_edit without history")
            prev = self.hist[rel].pop()
            if prev is None:
                raise Reject("undo of a create")
            p.write_text(prev)
            self.call(th, "write_file", {"filepath": rel, "content": self.fix(prev)}, ok(filepath=rel, size=len(self.fix(prev))))
            return
        raise Reject(f"unknown editor command {cmd}")

    @staticmethod
    def matches(lines, src):
        for n, t in src.items():
            if n == len(lines) + 1 and not t.strip():          # OpenHands shows the empty line after a final newline
                continue
            if n > len(lines) or lines[n - 1].lstrip("\ufeff").rstrip() != t.lstrip("\ufeff").rstrip():   # viewer drops a BOM
                return False
        return True

    def resync(self, p, rel, src_obs):
        """Write the file exactly as a full, unclipped source view shows it (the viewer splits on newlines, so a final
        empty line means a trailing newline)."""
        src = numbered(src_obs)
        if not src or sorted(src) != list(range(1, max(src) + 1)):
            raise Reject("viewed file missing in the checkout")
        p.parent.mkdir(parents=True, exist_ok=True)
        old = p.read_text(errors="replace") if p.exists() else None
        self.hist[rel].append(old)
        p.write_text("\n".join(src[n] for n in range(1, max(src) + 1)))

    @staticmethod
    def insert_anchor(content, line, new_str):
        """An insert-after-line as an edit_file: the anchor grows until it is unique in the file."""
        lines = content.split("\n")
        if line <= 0:
            for n in range(1, len(lines) + 1):
                anchor = "\n".join(lines[:n])
                if content.count(anchor) == 1:
                    return anchor, new_str + "\n" + anchor
        else:
            for n in range(1, line + 1):
                anchor = "\n".join(lines[line - n:line])
                if anchor.strip() and content.count(anchor) == 1:
                    return anchor, anchor + "\n" + new_str
        raise Reject("insert anchor not unique")


def user_prompt(repo, ps, hints, tree):
    parts = [f"You are evaluating a software engineering task for repository {repo}.\n\nProblem Statement:\n{ps}\n"]
    if hints.strip():
        parts.append(f"## Hints:\n{hints.strip()}\n")
    parts += [PROMPT_TAIL.split("\n\n## Execution")[0] + "\n", "## Execution" + PROMPT_TAIL.split("\n\n## Execution")[1].split("\n\n## Instructions:")[0] + "\n",
              "## Instructions:" + PROMPT_TAIL.split("\n\n## Instructions:")[1]]
    if tree:
        parts.append("## Workspace Layout\nThe repository is located at `/workspace`. Here is the directory tree (up to 3 levels):\n```\n" + tree + "\n```\n")
    return "\n".join(parts)


def touched_tests(msgs):
    return sorted({tc["function"]["arguments"].get("filepath") for m in msgs for tc in m.get("tool_calls") or []
                   if tc["function"]["name"] in ("edit_file", "write_file") and TEST_PATH.search(str(tc["function"]["arguments"].get("filepath")))})


def process_task(task, eps, cache, system_prompt):
    """Worker: one checkout, every trajectory of the task replayed on a fresh reset. -> (rows, rejects)."""
    rows, rejects = [], []
    ws = Path(tempfile.mkdtemp(prefix="ws_", dir=os.environ.get("CONVERT_TMP")))
    shutil.rmtree(ws)
    try:
        try:
            tree = prepare(task, cache, ws)
        except Exception as e:
            return [], [(ep["src"], ep["traj_id"], f"checkout failed: {str(e)[:200]}") for ep in eps]
        for ep in eps:
            sh(GIT + ["reset", "-q", "--hard", "HEAD"], cwd=ws, check=False)
            sh(GIT + ["clean", "-fdxq"], cwd=ws, check=False)
            try:
                ps = ep["problem"] or task["problem"]
                if not ps:
                    raise Reject("no problem statement")
                rp = Replay(ep, task, ws, tree)
                body = rp.run()
                user = user_prompt(task["repo"], rp.fix(ps), task.get("hints") or "", tree)
                msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}] + body
                lab = dict(rp.labels)
                lab["test_files_written"] = touched_tests(body)
                lab["n_tool_calls"] = sum(1 for m in body if m.get("tool_calls"))
                if ep["src_patch"]:
                    lab["final_diff_matches_source"] = norm_diff(rp.final_diff) == norm_diff(ep["src_patch"])
                rows.append({"instance_id": task["instance_id"], "repo": task["repo"], "source": f"external:{ep['src']}",
                             "source_dataset": SOURCES[ep["src"]][0], "source_traj_id": ep["traj_id"], "source_model": ep["model"],
                             "scaffold": ep["scaffold"], "source_resolved": True,
                             "base": {k: v for k, v in task["base"].items() if not k.endswith("_patch")},   # full recipe in tasks.jsonl
                             "labels": lab, "n_turns": lab["n_tool_calls"], "n_chars": sum(len(json.dumps(m)) for m in msgs),
                             "messages": msgs})
            except Reject as e:
                rejects.append((ep["src"], ep["traj_id"], str(e)))
            except Exception as e:
                rejects.append((ep["src"], ep["traj_id"], f"replay error {type(e).__name__}: {str(e)[:160]}"))
    finally:
        shutil.rmtree(ws, ignore_errors=True)
        d = Repo(cache, task["base"]["url"]).dir
        if d.exists():                              # never clone here: a failed clone already rejected the task
            sh(GIT + ["-C", str(d), "worktree", "prune"], check=False)
    return rows, rejects


def norm_diff(d):
    """Hunks of every pre-existing file the diff modifies (new files = scratch scripts / nebius's issue.md, ignored):
    the comparable part of two diffs."""
    files = {}
    for block in re.split(r"(?m)^diff --git ", d or "")[1:]:
        lines = block.split("\n")
        if any(l.startswith("new file mode") for l in lines[:4]):
            continue
        files[lines[0].split(" b/")[-1]] = "\n".join(l for l in lines[1:] if not l.startswith("index ")).strip()
    return files


# ---------------------------------------------------------------- driver
def units(src, max_rows):
    """Yield (unit_name, parquet file, row_group) for a source."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    ds, glob = SOURCES[src][:2]
    for f in sorted(fs.glob(f"datasets/{ds}/{glob}")):
        n = pq.ParquetFile(fs.open(f, block_size=1 << 20)).num_row_groups
        for g in range(n):
            yield f"{Path(f).stem}-rg{g:03d}", f, g


def read_unit(src, f, g):
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    pf = pq.ParquetFile(HfFileSystem().open(f, block_size=1 << 24))
    for b in pf.iter_batches(row_groups=[g], batch_size=128):
        for r in b.to_pylist():
            if src == "rebench_oh" and not r.get("resolved"):
                continue
            if src == "smith" and not r.get("resolved"):
                continue
            yield r


def evict(cache, cap_gb):
    reps = [p for p in Path(cache).iterdir() if p.is_dir() and not p.name.endswith(".tmp")]
    size = lambda p: sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    sizes = {p: size(p) for p in reps}
    total = sum(sizes.values())
    for p in sorted(reps, key=lambda p: float((p / "LAST_USED").read_text()) if (p / "LAST_USED").exists() else 0):
        if total <= cap_gb * 2 ** 30:
            break
        shutil.rmtree(p, ignore_errors=True)
        total -= sizes[p]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--comp-tasks", required=True)
    ap.add_argument("--system-prompt", required=True, help="file: the deployed system prompt")
    ap.add_argument("--sources", default="coderforge,rebench_oh")
    ap.add_argument("--models", default="Qwen3-Coder-480B-A35B-Instruct",
                    help="keep only trajectories whose source model is stronger than Gemma 4 31B (comma list; 'any' = no filter)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--cache-gb", type=float, default=8)
    ap.add_argument("--limit-units", type=int, default=0, help="per source (smoke test)")
    ap.add_argument("--max-rows", type=int, default=0, help="per unit (smoke test)")
    a = ap.parse_args()
    out = Path(a.out)
    cache = Path(a.cache or out.parent / "git_cache")
    cache.mkdir(parents=True, exist_ok=True)
    system_prompt = Path(a.system_prompt).read_text()
    vf = out / "VERSION"
    if not vf.exists() or vf.read_text().strip() != VERSION:
        for d in ("train", "rejects"):
            shutil.rmtree(out / d, ignore_errors=True)
        (out / "tasks.jsonl").unlink(missing_ok=True)
        out.mkdir(parents=True, exist_ok=True)
        vf.write_text(VERSION)
    comp = set()
    for l in open(a.comp_tasks):
        if l.strip():
            iid = json.loads(l)["instance_id"]
            m = re.match(r"(\w+?)[_-](\d+)$", iid.split("__")[-1])
            if m:
                comp.add((m.group(1).lower(), m.group(2)))
    resolver = Resolver(load_meta(out.parent), comp)
    print(f"meta: {len(resolver.reb)} SWE-rebench, {len(resolver.smith)} SWE-smith tasks; {len(comp)} competition PRs excluded", flush=True)
    for src in a.sources.split(","):
        (out / "train" / src).mkdir(parents=True, exist_ok=True)
        (out / "rejects" / src).mkdir(parents=True, exist_ok=True)
        for ui, (uname, f, g) in enumerate(units(src, a.max_rows)):
            if a.limit_units and ui >= a.limit_units:
                break
            dst = out / "train" / src / f"{uname}.jsonl.gz"
            if dst.exists():
                continue
            t0, groups, tasks, rej, n_kept, seen = time.time(), defaultdict(list), {}, [], 0, set()
            tmp = dst.with_suffix(".tmp")
            fo = gzip.open(tmp, "wt")

            def flush():                            # convert the buffered tasks; rows go straight to the shard
                nonlocal n_kept
                with ProcessPoolExecutor(a.workers) as ex:
                    futs = {ex.submit(process_task, tasks[k], v, str(cache), system_prompt): v for k, v in groups.items()}
                    for fu in as_completed(futs):
                        try:
                            rr, jj = fu.result()
                        except Exception as e:      # a crashed worker rejects its task, never the whole job
                            rr, jj = [], [(ep["src"], ep["traj_id"], f"worker error {type(e).__name__}: {str(e)[:160]}") for ep in futs[fu]]
                        for r in rr:
                            fo.write(json.dumps(r) + "\n")
                        n_kept += len(rr)
                        rej.extend(jj)
                seen.update(groups)
                groups.clear()

            n_eps = 0
            for i, r in enumerate(read_unit(src, f, g)):
                if a.max_rows and i >= a.max_rows:
                    break
                try:
                    ep = normalize(src, r)
                    if a.models != "any" and (ep["model"] or "unknown").split("/")[-1] not in a.models.split(","):
                        raise Reject(f"model not stronger than Gemma 4 31B ({ep['model']})")
                    t = resolver.task_for(ep)
                    tasks[t["instance_id"]] = t
                    groups[t["instance_id"]].append(ep)
                    n_eps += 1
                    if n_eps % 500 == 0:
                        flush()
                except Reject as e:
                    rej.append((src, str(r.get("trajectory_id") or r.get("traj_id") or r.get("instance_id")), str(e)))
            flush()
            fo.close()
            with gzip.open(out / "rejects" / src / f"{uname}.jsonl.gz", "wt") as fr:
                for s_, t_, why_ in rej:
                    fr.write(json.dumps({"source": s_, "traj_id": t_, "reason": why_}) + "\n")
            with open(out / "tasks.jsonl", "a") as ft:
                for k in sorted(seen):
                    t = tasks[k]
                    ft.write(json.dumps({"instance_id": k, "repo": t["repo"], "problem_statement": t["problem"], "hints_text": t.get("hints") or "",
                                         "base": t["base"], "source": src}) + "\n")
            tmp.rename(dst)
            why = Counter(re.sub(r"\(.*|: .*", "", x[2]).strip() for x in rej)
            print(f"[{src} {uname}] kept {n_kept} / {n_kept + len(rej)} trajectories over {len(seen)} tasks in "
                  f"{time.time() - t0:.0f}s; rejects {dict(why.most_common(6))}", flush=True)
            evict(cache, a.cache_gb)
    manifest(out)


def manifest(out):
    kept, rej, tasks, labels = Counter(), defaultdict(Counter), defaultdict(set), defaultdict(Counter)
    for p in (out / "train").glob("*/*.jsonl.gz"):
        for l in gzip.open(p, "rt"):
            r = json.loads(l)
            s = r["source"].split(":")[1]
            kept[s] += 1
            tasks[s].add(r["instance_id"])
            for k, v in r["labels"].items():
                labels[s][k] += bool(v)
    for p in (out / "rejects").glob("*/*.jsonl.gz"):
        for l in gzip.open(p, "rt"):
            r = json.loads(l)
            rej[r["source"]][re.sub(r"\(.*|: .*", "", r["reason"]).strip()] += 1
    m = {s: {"dataset": SOURCES[s][0], "kept": kept[s], "tasks": len(tasks[s]), "rejected": sum(rej[s].values()),
             "reject_reasons": dict(rej[s].most_common()), "rows_with_label": dict(labels[s])} for s in SOURCES if kept[s] or rej[s]}
    (out / "manifest.json").write_text(json.dumps(m, indent=1))
    print(json.dumps(m, indent=1))


if __name__ == "__main__":
    main()
