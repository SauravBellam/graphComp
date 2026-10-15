#!/usr/bin/env python3
"""Synthetic SWE tasks: procedural bugs injected into competition snapshots (SWE-smith 2504.21798, "Procedural" strategy).

    python make_synthetic.py build  --comp COMP --out OUT --work WORK --tools TOOLS [--target fastapi=40,...] [--workers 6]
    python make_synthetic.py issues --out OUT --model MODEL_DIR
    python make_synthetic.py --selftest

build, per repo: take the snapshot of the newest competition task, run its test suite once under coverage with per-test
contexts (which tests execute which line), then try AST mutations (flip a comparison / and-or, negate or drop an if, drop a
statement, constant +1) on covered lines of the package and run only the covering tests. A mutation becomes a task if some
covering test that passed at baseline (in a test file with no baseline failures) now fails. Up to 3 failing test functions
from one file are removed from the snapshot and restored by test_patch, which is what the harness grades (it runs the
test_patch files and requires the added test functions to pass). patch = the revert of the mutation. The snapshot is a fresh
one-commit repo (same author / message as the source commit), so git history and git diff do not reveal the bug.
Never mutated: functions that any competition gold patch touches (leakage guard). base_commit stays the competition
commit, so the competition graph / embeddings are reused (copied into OUT; the mutation changes one expression).
Output = the external-task layout (tasks.jsonl, snapshots/, graphs/, embeddings/, graphcomp_external.json), which
notebook 20 picks up as one more source. Validity: run_arm.py gold / empty + import_external.py --record-validity.

issues: writes each task's problem statement with a local model (vLLM offline), R2E-Gym style (2504.07164): the model
sees the fix and the failing test output, writes a user-facing bug report; rejected if it names a hidden test or quotes a
changed line (SWE-smith: leaking the tests makes the solver skip reproduction). Tasks without an accepted issue are dropped.
"""
import argparse, ast, json, os, random, re, shutil, subprocess, sys, tarfile, time
import xml.etree.ElementTree as ET
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

PKG = {"fastapi": "fastapi", "rich": "rich", "requests": "requests", "httpx": "httpx"}
CMP = {ast.Eq: ("==", "!="), ast.NotEq: ("!=", "=="), ast.Lt: ("<", ">="), ast.LtE: ("<=", ">"), ast.Gt: (">", "<="),
       ast.GtE: (">=", "<"), ast.Is: (r"\bis\b", "is not"), ast.IsNot: (r"\bis\s+not\b", "is"),
       ast.In: (r"\bin\b", "not in"), ast.NotIn: (r"\bnot\s+in\b", "in")}
MAX_RUN_TESTS, RUN_TIMEOUT, MAX_HIDE = 40, 240, 3


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ---------------------------------------------------------------- mutations (pure functions of the source text)
def _offsets(src):
    lines = src.splitlines(keepends=True)
    starts, s = [], 0
    for l in lines:
        starts.append(s); s += len(l)
    def off(lineno, col_bytes):
        line = lines[lineno - 1]
        return starts[lineno - 1] + len(line.encode()[:col_bytes].decode(errors="ignore"))
    return lines, starts, off


def _owns_lines(lines, node):
    """The statement starts its first line and ends its last one (no `a = 1; b = 2`)."""
    before = lines[node.lineno - 1].encode()[:node.col_offset].decode(errors="ignore")
    after = lines[node.end_lineno - 1].encode()[node.end_col_offset:].decode(errors="ignore").strip()
    return not before.strip() and (not after or after.startswith("#"))


def mutations(src):
    """Yield (op, lineno, end_lineno, function_name, new_src) for every mutation site."""
    tree = ast.parse(src)
    lines, starts, off = _offsets(src)
    seg = lambda n: (off(n.lineno, n.col_offset), off(n.end_lineno, n.end_col_offset))

    def drop_lines(n):
        a = starts[n.lineno - 1]
        b = starts[n.end_lineno - 1] + len(lines[n.end_lineno - 1])
        return src[:a] + src[b:]

    def visit(fn):
        # dropping the only binding of a local turns the bug into a NameError (trivial): keep only re-bound names
        stores = defaultdict(int)
        for n in ast.walk(fn):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                stores[n.id] += 1
        params = {p.arg for p in ast.walk(fn.args) if isinstance(p, ast.arg)}
        rebound = lambda st: not isinstance(st, ast.Assign) or all(
            not isinstance(t, ast.Name) or stores[t.id] > 1 or t.id in params for t in st.targets)
        for parent in ast.walk(fn):
            for field in ("body", "orelse", "finalbody"):
                body = getattr(parent, field, None)
                if not isinstance(body, list) or len(body) < 2:
                    continue
                for st in body:
                    if not _owns_lines(lines, st):
                        continue
                    if isinstance(st, ast.If) and not st.orelse:
                        yield "drop_if", st, drop_lines(st)
                    elif (isinstance(st, (ast.Assign, ast.AugAssign)) and rebound(st)) or \
                            (isinstance(st, ast.Expr) and isinstance(st.value, ast.Call)):
                        yield "drop_stmt", st, drop_lines(st)
        for n in ast.walk(fn):
            if isinstance(n, ast.Compare) and len(n.ops) == 1 and type(n.ops[0]) in CMP:
                pat, rep = CMP[type(n.ops[0])]
                a, b = off(n.left.end_lineno, n.left.end_col_offset), off(n.comparators[0].lineno, n.comparators[0].col_offset)
                mid, k = re.subn(pat if pat.startswith("\\") else re.escape(pat), rep, src[a:b], count=1)
                if k:
                    yield "flip_compare", n, src[:a] + mid + src[b:]
            elif isinstance(n, ast.BoolOp) and len(n.values) == 2:
                a, b = off(n.values[0].end_lineno, n.values[0].end_col_offset), off(n.values[1].lineno, n.values[1].col_offset)
                old, new = ("and", "or") if isinstance(n.op, ast.And) else ("or", "and")
                mid, k = re.subn(rf"\b{old}\b", new, src[a:b], count=1)
                if k:
                    yield "flip_boolop", n, src[:a] + mid + src[b:]
            elif isinstance(n, ast.If):
                a, b = seg(n.test)
                if isinstance(n.test, ast.UnaryOp) and isinstance(n.test.op, ast.Not):
                    oa, ob = seg(n.test.operand)
                    new = src[oa:ob]
                else:
                    new = f"not ({src[a:b]})"
                yield "negate_if", n.test, src[:a] + new + src[b:]
            elif isinstance(n, (ast.Compare, ast.BinOp, ast.Subscript, ast.Slice)):
                for c in ast.iter_child_nodes(n):
                    if isinstance(c, ast.Constant) and type(c.value) is int and c.value < 1000:
                        a, b = seg(c)
                        yield "const_plus_one", c, src[:a] + str(c.value + 1) + src[b:]

    seen = set()
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for op, node, new in visit(fn):
                key = (op, node.lineno, node.col_offset, new)
                if key in seen or new == src:
                    continue
                seen.add(key)
                try:
                    ast.parse(new)
                except SyntaxError:
                    continue
                yield op, node.lineno, node.end_lineno, fn.name, new


def remove_test_functions(src, targets):
    """targets: set of (class_name or '', function_name). Returns new source or None if a class body would become empty."""
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    drop = []
    for scope, body in [("", tree.body)] + [(c.name, c.body) for c in tree.body if isinstance(c, ast.ClassDef)]:
        hit = [n for n in body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and (scope, n.name) in targets]
        if scope and hit and len(hit) == len([n for n in body if not isinstance(n, ast.Expr)]):
            return None
        for n in hit:
            drop.append((min([n.lineno] + [d.lineno for d in n.decorator_list]), n.end_lineno))
    if len(drop) != len(targets):
        return None
    keep = [l for i, l in enumerate(lines, 1) if not any(a <= i <= b for a, b in drop)]
    return "".join(keep)


def touched_functions(patches):
    """{file: {function or class names}} touched by competition gold patches (hunk headers + changed def lines)."""
    out = defaultdict(set)
    for p in patches:
        f = None
        for l in p.splitlines():
            if l.startswith("+++ b/"):
                f = l[6:]
            elif f and l.startswith("@@"):
                m = re.search(r"@@.*@@\s*(?:async\s+)?(?:def|class)\s+(\w+)", l)
                if m: out[f].add(m.group(1))
            elif f and l[:1] in "+-":
                m = re.match(r"[+-]\s*(?:async\s+)?def\s+(\w+)", l)
                if m: out[f].add(m.group(1))
    return out


# ---------------------------------------------------------------- running tests
def env_for(root, a):
    pp = [a.tools, str(Path(root) / "src"), str(root)] + ([os.environ["SWE_EXTRA_PYTHONPATH"]] if os.environ.get("SWE_EXTRA_PYTHONPATH") else [])
    return dict(os.environ, PYTHONPATH=":".join(pp), PYTHONDONTWRITEBYTECODE="1")


def parse_junit(path):
    """{(file, class, function): (status, message)}, worst case over parametrizations. status: pass / skip / error (setup or
    teardown, e.g. a fixture's port already in use: never evidence of the bug) / fail (the test body failed)."""
    res = {}
    rank = {"pass": 0, "skip": 1, "error": 2, "fail": 3}
    for tc in ET.parse(path).getroot().iter("testcase"):
        f = tc.attrib.get("file", "")
        mod = f[:-3].replace("/", ".") if f.endswith(".py") else ""
        cls = tc.attrib.get("classname", "")
        cls = cls[len(mod) + 1:] if mod and cls.startswith(mod + ".") else ("" if cls == mod else cls.rsplit(".", 1)[-1])
        name = tc.attrib.get("name", "").split("[", 1)[0]
        bad = tc.find("failure") if tc.find("failure") is not None else tc.find("error")
        st = ("fail" if bad.tag == "failure" else "error") if bad is not None else "skip" if tc.find("skipped") is not None else "pass"
        msg = ((bad.attrib.get("message", "") + "\n" + (bad.text or "")) if bad is not None else "")[-2000:]
        k = (f, cls, name)
        if k not in res or rank[st] > rank[res[k][0]]:
            res[k] = (st, msg)
    return res


def nodeid(k):
    f, cls, name = k
    return f"{f}::{cls}::{name}" if cls else f"{f}::{name}"


_W = {}


def _winit(base, wdir, a_dict, q):
    i = q.get()
    d = Path(wdir) / f"w{i}"
    if not d.exists():
        shutil.copytree(base, d, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    _W.update(dir=d, a=argparse.Namespace(**a_dict), base=Path(base))


def _wrun(job):
    cid, rel, new_src, tests = job
    d, a = _W["dir"], _W["a"]
    p = d / rel
    old = p.read_text()
    try:
        p.write_text(new_src)
        x = d / f".junit_{cid}.xml"
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o", "junit_family=xunit1",
                            f"--junitxml={x}", *tests], cwd=d, env=env_for(d, a), capture_output=True, timeout=RUN_TIMEOUT)
        res = parse_junit(x) if x.exists() else None
        x.unlink(missing_ok=True)
        return cid, r.returncode, res
    except subprocess.TimeoutExpired:
        return cid, "timeout", None
    finally:
        p.write_text(old)


# ---------------------------------------------------------------- build
def git(d, *args, env=None, check=True):
    return subprocess.run(["git", "-C", str(d), *args], capture_output=True, text=True, check=check, env=env).stdout


def build_repo(short, comp_rows, a, out, rows_out, meta_out, rng):
    rows = sorted([r for r in comp_rows if r["instance_id"].split("_")[0] == short], key=lambda r: r["created_at"])
    src_task = rows[-1]
    work = Path(a.work) / short
    shutil.rmtree(work, ignore_errors=True)
    base = work / "base"
    base.mkdir(parents=True)
    with tarfile.open(Path(a.comp) / "snapshots" / f"{src_task['instance_id']}.tgz") as tf:
        tf.extractall(base, filter="tar") if sys.version_info >= (3, 12) else tf.extractall(base)
    pkg = next(p for p in (base / "src" / PKG[short], base / PKG[short]) if p.is_dir())
    log(f"{short}: base = {src_task['instance_id']} @ {src_task['base_commit'][:10]}, package {pkg.relative_to(base)}")

    # 1. baseline under coverage, per-test contexts
    rc = work / "covrc"
    rc.write_text(f"[run]\ndynamic_context = test_function\nsource = {pkg}\ndata_file = {work}/.coverage\n")
    t0 = time.time()
    bj = work / "baseline.xml"
    r = subprocess.run([sys.executable, "-m", "coverage", "run", f"--rcfile={rc}", "-m", "pytest", "-q", "-p", "no:cacheprovider",
                        "-o", "junit_family=xunit1", f"--junitxml={bj}", "tests"], cwd=base, env=env_for(base, a),
                       capture_output=True, text=True, timeout=3600)
    if not bj.exists():
        log(f"{short}: baseline produced no junit (rc {r.returncode}); skipping repo\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
        return 0
    subprocess.run([sys.executable, "-m", "coverage", "json", f"--rcfile={rc}", "--show-contexts", "-o", str(work / "cov.json")],
                   cwd=base, env=env_for(base, a), check=True, capture_output=True)
    baseline = parse_junit(bj)
    bad_files = {k[0] for k, (st, _) in baseline.items() if st in ("fail", "error")}
    eligible = {k for k, (st, _) in baseline.items() if st == "pass" and k[0] not in bad_files}
    log(f"{short}: baseline {len(baseline)} test functions in {time.time() - t0:.0f}s; {len(bad_files)} test files with failures "
        f"(excluded); {len(eligible)} eligible F2P tests")
    # coverage context -> junit key: contexts are "<module>.<Class>.<func>"; match on the dotted suffix
    by_dotted = {}
    for k in baseline:
        f, cls, name = k
        by_dotted[".".join(([f[:-3].replace("/", ".")] if f else []) + ([cls] if cls else []) + [name])] = k
    memo = {}
    def ctx_key(c):
        if c not in memo:
            s = c.split("|")[0]
            hits = [by_dotted[s]] if s in by_dotted else \
                [k for d, k in by_dotted.items() if d.endswith("." + s) or s.endswith("." + d)]
            memo[c] = hits[0] if len(hits) == 1 else None
        return memo[c]
    cov = json.load(open(work / "cov.json"))["files"]

    # 2. candidates
    touched = touched_functions([r["patch"] for r in rows])
    by_op = defaultdict(list)
    n_guard = 0
    for fpath, info in cov.items():
        rel = str(Path(fpath).resolve().relative_to(base.resolve())) if Path(fpath).is_absolute() else fpath
        ctxs = info.get("contexts", {})
        try:
            src = (base / rel).read_text()
            muts = list(mutations(src))
        except Exception:
            continue
        for op, l0, l1, fn, new in muts:
            if fn in touched.get(rel, ()):
                n_guard += 1; continue
            tests = {ctx_key(c) for ln in range(l0, l1 + 1) for c in ctxs.get(str(ln), []) if c}
            tests = sorted(t for t in tests if t in eligible)
            if tests:
                by_op[op].append((rel, fn, op, l0, new, tests))
    for v in by_op.values():
        rng.shuffle(v)
    order = [c for grp in zip(*by_op.values()) for c in grp] if by_op else []
    order += [c for v in by_op.values() for c in v[min(map(len, by_op.values())):]]
    log(f"{short}: {sum(map(len, by_op.values()))} covered mutation sites "
        f"({', '.join(f'{k} {len(v)}' for k, v in sorted(by_op.items()))}); {n_guard} skipped by the competition-patch guard")

    # 3. evaluate in parallel until the target is met
    target = a.target[short]
    jobs = []
    for i, (rel, fn, op, l0, new, tests) in enumerate(order[: target * a.max_tries]):
        run = tests if len(tests) <= MAX_RUN_TESTS else rng.sample(tests, MAX_RUN_TESTS)
        jobs.append((i, rel, new, [nodeid(t) for t in run]))
    from multiprocessing import Manager
    q = Manager().Queue()
    for w in range(a.workers):
        q.put(w)
    accepted, per_file, per_fn, stats = [], defaultdict(int), set(), defaultdict(int)
    with Pool(a.workers, initializer=_winit, initargs=(str(base), str(work), vars(a), q)) as pool:
        for cid, rc_, res in pool.imap_unordered(_wrun, jobs):
            rel, fn, op, l0, new, tests = order[cid]
            if res is None:
                stats["timeout_or_no_junit"] += 1; continue
            fails = [k for k, (st, _) in res.items() if st == "fail" and k in eligible]
            if not fails:
                stats["survived"] += 1; continue
            if (rel, fn) in per_fn or per_file[rel] >= 4:
                stats["diversity_cap"] += 1; continue
            by_file = defaultdict(list)
            for k in fails:
                by_file[k[0]].append(k)
            tfile = max(sorted(by_file), key=lambda f: len(by_file[f]))
            hide = sorted(by_file[tfile])[:MAX_HIDE]
            tsrc = (base / tfile).read_text()
            if remove_test_functions(tsrc, {(k[1], k[2]) for k in hide}) is None:
                stats["unhideable_tests"] += 1; continue
            per_fn.add((rel, fn)); per_file[rel] += 1
            accepted.append(dict(rel=rel, function=fn, op=op, line=l0, new=new, test_file=tfile, hide=hide,
                                 n_run=len(res), n_fail=len([k for k, (st, _) in res.items() if st != "pass"]),
                                 fail_msgs={nodeid(k): res[k][1] for k in hide}))
            stats["accepted"] += 1
            if len(accepted) >= target:
                pool.terminate(); break
    log(f"{short}: tried {sum(stats.values())} mutations -> {dict(stats)}")

    # 4. materialize tasks
    head = git(base, "log", "-1", "--format=%an%n%ae%n%aI%n%B").split("\n", 3)
    genv = dict(os.environ, GIT_AUTHOR_NAME=head[0], GIT_AUTHOR_EMAIL=head[1], GIT_AUTHOR_DATE=head[2],
                GIT_COMMITTER_NAME=head[0], GIT_COMMITTER_EMAIL=head[1], GIT_COMMITTER_DATE=head[2])
    n = 0
    for j, c in enumerate(accepted):
        iid = f"{short}_syn{j:03d}"
        snap = work / "snap"
        shutil.rmtree(snap, ignore_errors=True)
        shutil.copytree(base, snap, ignore=shutil.ignore_patterns(".git", "__pycache__"), symlinks=True)
        orig_src, orig_test = (base / c["rel"]).read_text(), (base / c["test_file"]).read_text()
        (snap / c["rel"]).write_text(c["new"])
        (snap / c["test_file"]).write_text(remove_test_functions(orig_test, {(k[1], k[2]) for k in c["hide"]}))
        git(snap, "init", "-q"); git(snap, "add", "-A")
        git(snap, "commit", "-q", "-m", head[3].strip() or "update", env=genv)
        (snap / c["rel"]).write_text(orig_src); patch = git(snap, "diff", "--", c["rel"]); git(snap, "checkout", "-q", "--", c["rel"])
        (snap / c["test_file"]).write_text(orig_test); tpatch = git(snap, "diff", "--", c["test_file"]); git(snap, "checkout", "-q", "--", c["test_file"])
        with tarfile.open(out / "snapshots" / f"{iid}.tgz", "w:gz") as tf:
            tf.add(snap, arcname=".")
        shutil.rmtree(snap)
        rows_out.append({"instance_id": iid, "repo": src_task["repo"], "base_commit": src_task["base_commit"], "problem_statement": "",
                         "hints_text": "", "patch": patch, "test_patch": tpatch, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ")})
        meta_out.append({"instance_id": iid, "source_task": src_task["instance_id"], "op": c["op"], "file": c["rel"],
                         "function": c["function"], "line": c["line"], "test_file": c["test_file"],
                         "hidden_tests": [nodeid(k) for k in c["hide"]], "n_tests_run": c["n_run"], "n_tests_failed": c["n_fail"],
                         "fail_msgs": c["fail_msgs"]})
        n += 1
    for d in ("graphs", "embeddings"):
        ext = "json" if d == "graphs" else "npz"
        g = Path(a.comp) / d / f"{short}_{src_task['base_commit']}.{ext}"
        if g.exists():
            shutil.copy(g, out / d / g.name)
        else:
            log(f"{short}: WARNING no competition {d[:-1]} for {g.name}")
    shutil.rmtree(work, ignore_errors=True)
    log(f"{short}: {n} tasks written")
    return n


def cmd_build(a):
    out = Path(a.out)
    for d in ("snapshots", "graphs", "embeddings"):
        (out / d).mkdir(parents=True, exist_ok=True)
    comp_rows = [json.loads(l) for l in open(Path(a.comp) / "tasks.jsonl") if l.strip()]
    rng = random.Random(a.seed)
    rows, meta = [], []
    for short in a.target:
        try:
            build_repo(short, comp_rows, a, out, rows, meta, rng)
        except Exception as e:
            log(f"{short}: FAILED {type(e).__name__}: {e}")
    with open(out / "tasks.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    with open(out / "synthetic_meta.jsonl", "w") as f:
        f.writelines(json.dumps(m) + "\n" for m in meta)
    (out / "graphcomp_external.json").write_text(json.dumps({
        "built": [r["instance_id"] for r in rows], "valid": [], "kind": "synthetic procedural bugs (make_synthetic.py)",
        "seed": a.seed, "target": a.target, "note": "valid is filled by import_external.py --record-validity"}, indent=1))
    log(f"built {len(rows)} synthetic tasks into {out}")
    if not rows:
        sys.exit("no tasks built")


# ---------------------------------------------------------------- issues
PROMPT = """You are writing a GitHub issue for the {repo} project. A bug exists in the current code. Below is the fix the maintainers will eventually apply (do NOT reveal it) and the output of the project's tests that fail because of the bug.

Write the issue as a user who hit the bug would: a short title line, what they did, what they expected, and what happened instead, with a small reproduction script that uses the public API (it may be approximate). Do not mention the fix, line numbers, the diff, test files or test function names, and do not quote the changed lines. Output only the issue text.

FIX (hidden from the reader):
{patch}

FAILING TEST OUTPUT:
{fails}
"""


def issue_ok(text, row, meta):
    if not 200 <= len(text) <= 4000 or "@@" in text or "diff --git" in text:
        return False
    names = {t.split("::")[-1] for t in meta["hidden_tests"]}
    if any(n in text for n in names):
        return False
    changed = [l[1:].strip() for l in row["patch"].splitlines() if l[:1] in "+-" and not l.startswith(("+++", "---"))]
    return not any(len(c) >= 12 and c in text for c in changed)


def cmd_issues(a):
    from vllm import LLM, SamplingParams
    out = Path(a.out)
    meta_g = json.loads((out / "graphcomp_external.json").read_text())
    rows = [json.loads(l) for l in open(out / "tasks.jsonl") if l.strip()]
    meta = {m["instance_id"]: m for m in map(json.loads, open(out / "synthetic_meta.jsonl"))}
    todo = [r for r in rows if r["instance_id"] in set(meta_g.get("valid") or [])]
    log(f"issues for {len(todo)} valid tasks with {a.model}")
    llm = LLM(model=a.model, max_model_len=8192, gpu_memory_utilization=0.85, seed=a.seed)
    sp = SamplingParams(n=4, temperature=0.8, top_p=0.95, max_tokens=900, seed=a.seed)
    msgs = [[{"role": "user", "content": PROMPT.format(repo=r["repo"], patch=r["patch"][:4000],
                                                         fails="\n\n".join(f"{k}:\n{v[-1500:]}" for k, v in meta[r["instance_id"]]["fail_msgs"].items())[:5000])}]
            for r in todo]
    outs = llm.chat(msgs, sp, chat_template_kwargs={"enable_thinking": False})
    kept, rejected = [], []
    for r, o in zip(todo, outs):
        good = [c.text.strip() for c in o.outputs if issue_ok(c.text.strip(), r, meta[r["instance_id"]])]
        if good:
            r["problem_statement"] = good[0]; kept.append(r)
        else:
            rejected.append(r["instance_id"])
    with open(out / "tasks.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in kept)
    keep_ids = {r["instance_id"] for r in kept}
    for p in (out / "snapshots").glob("*.tgz"):
        if p.stem not in keep_ids:
            p.unlink()
    meta_g.update(valid=sorted(keep_ids), issue_model=a.model, issue_rejected=rejected)
    (out / "graphcomp_external.json").write_text(json.dumps(meta_g, indent=1))
    log(f"issues: kept {len(kept)}, rejected {len(rejected)} (no sample passed the leak filter) {rejected}")
    for r in kept[:3]:
        print(f"---- {r['instance_id']}\n{r['problem_statement']}\n", flush=True)


# ---------------------------------------------------------------- self-test
def selftest():
    src = ("def f(x, y):\n    z = x + 1\n    if x == y and y > 2:\n        z = 0\n    print(z)\n    return z if x[1:3] else None\n\n"
           "class C:\n    def g(self, a):\n        a.b = 3; return a\n")
    got = list(mutations(src))
    ops = {o for o, *_ in got}
    assert {"flip_compare", "flip_boolop", "negate_if", "drop_if", "drop_stmt", "const_plus_one"} <= ops, ops
    assert all(fn == "f" for *_, fn, _ in got), "g's `a.b = 3; return a` shares a line and has one statement per line-owner"
    for op, *_, new in got:
        ast.parse(new); assert new != src
    flips = [n for o, *_, n in got if o == "flip_compare"]
    assert any("x != y" in n for n in flips) and any("y <= 2" in n for n in flips)
    assert any("x == y or y > 2" in n for o, *_, n in got if o == "flip_boolop")
    assert any("if not (x == y and y > 2):" in n for o, *_, n in got if o == "negate_if")
    t = "import pytest\n\n@pytest.mark.x\ndef test_a():\n    assert 1\n\ndef test_b():\n    pass\n\nclass TestK:\n    def test_c(self):\n        pass\n"
    r = remove_test_functions(t, {("", "test_a")})
    assert "test_a" not in r and "@pytest.mark.x" not in r and "test_b" in r
    assert remove_test_functions(t, {("TestK", "test_c")}) is None
    assert remove_test_functions(t, {("", "test_zz")}) is None
    tf = touched_functions(["+++ b/pkg/m.py\n@@ -1,2 +1,2 @@ def foo(a):\n-    x\n+    y\n+def bar():\n"])
    assert tf == {"pkg/m.py": {"foo", "bar"}}, tf
    row = {"patch": "--- a/m.py\n+++ b/m.py\n-    if left != right:\n+    if left == right:\n"}
    m = {"hidden_tests": ["tests/t.py::test_eq"]}
    assert not issue_ok("x" * 300 + "test_eq", row, m) and not issue_ok("x" * 300 + "if left == right:", row, m)
    assert issue_ok("Calling f with equal values returns the wrong branch. " * 6, row, m)
    ne = "def h(a):\n    m = a + 1\n    print(m)\n    k = 0\n    k = a\n    return k\n"
    dropped = [n for o, *_, n in mutations(ne) if o == "drop_stmt"]
    assert not any("m = a + 1" not in n for n in dropped) and any(n.count("k =") == 1 for n in dropped), dropped
    x = ET.fromstring('<testsuite><testcase file="t.py" classname="t" name="test_a[1]"><error message="port"/></testcase>'
                      '<testcase file="t.py" classname="t" name="test_a[2]"/><testcase file="t.py" classname="t.K" name="test_b">'
                      '<failure message="assert"/></testcase></testsuite>')
    Path("/tmp/_junit_selftest.xml").write_bytes(ET.tostring(x))
    j = parse_junit("/tmp/_junit_selftest.xml")
    assert j[("t.py", "", "test_a")][0] == "error" and j[("t.py", "K", "test_b")][0] == "fail", j
    print("selftest ok")


def main():
    if "--selftest" in sys.argv:
        return selftest()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--comp", required=True); b.add_argument("--out", required=True); b.add_argument("--work", required=True)
    b.add_argument("--tools", required=True, help="dir with `coverage` (pip --target), first on PYTHONPATH")
    b.add_argument("--target", default="fastapi=40,rich=30,requests=15,httpx=15",
                   type=lambda s: {k: int(v) for k, v in (p.split("=") for p in s.split(","))})
    b.add_argument("--workers", type=int, default=6); b.add_argument("--max-tries", type=int, default=8)
    b.add_argument("--seed", type=int, default=1)
    i = sub.add_parser("issues")
    i.add_argument("--out", required=True); i.add_argument("--model", required=True); i.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    cmd_build(a) if a.cmd == "build" else cmd_issues(a)


if __name__ == "__main__":
    main()
