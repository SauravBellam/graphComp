#!/usr/bin/env python3
"""Synthetic SWE tasks on the competition repos, SWE-smith style (2504.21798): procedural bugs, combined bugs, PR mirrors.

    python make_synthetic.py build  --comp COMP --out OUT --work WORK --tools TOOLS --guard guard_tasks.json [--target ...]
    python make_synthetic.py issues --out OUT --model MODEL_DIR
    python make_synthetic.py --selftest

build, per repo, at the snapshot of the newest competition task:
  1. baseline: the test suite once under coverage with per-test contexts (which tests execute which line).
  2. single bugs (procedural): AST mutations on covered package lines (flip a comparison / and-or / + - / True False,
     negate or drop an if, swap if-else bodies, drop a loop or a statement). Kept if a covering test that passed at
     baseline now fails in its body (setup errors, e.g. a port in use, never count). One bug per function.
  3. combined bugs (SWE-smith B.3): 2-4 kept single bugs from one file (<= 3 combos per file), or 2-3 from different files
     of one module (= package subdirectory; <= 6 per module); every single bug is used in at most one combo of each kind;
     the merged bug is re-run and kept only if it still fails tests.
  4. PR mirror (SWE-smith's best strategy): undo a past commit's package changes at the snapshot (clean `git apply -R`,
     <= 150 changed lines), kept if tests that passed at baseline now fail; the fix is the commit's own change.
Never used: functions touched by the gold patch of a guarded task (dev / holdout slices, --guard) and commits that carry a
competition or imported PR number (the mirror would recreate that task). A kept bug becomes a task: up to 5 failing test
functions (one per bug first, <= 3 files) are removed from the snapshot and restored by test_patch, which is what the
harness grades; patch = the fix. The snapshot is a fresh one-commit repo (same author / message), so git does not show the
bug. base_commit stays the competition commit (competition graph / embeddings reused). Output = the external-task layout
(tasks.jsonl, snapshots/, graphs/, embeddings/, graphcomp_external.json) that notebook 20 loads as one more source.

issues: problem statements from a local model (vLLM offline), R2E-Gym style (2504.07164): the model sees the fix and the
failing test output and writes a user's bug report; rejected if it names a hidden test or quotes a changed line.
"""
import argparse, ast, itertools, json, os, random, re, shutil, subprocess, sys, tarfile, tempfile, time
import xml.etree.ElementTree as ET
from collections import defaultdict
from multiprocessing import Manager, Pool
from pathlib import Path

PKG = {"fastapi": "fastapi", "rich": "rich", "requests": "requests", "httpx": "httpx"}
CMP = {ast.Eq: ("==", "!="), ast.NotEq: ("!=", "=="), ast.Lt: ("<", ">="), ast.LtE: ("<=", ">"), ast.Gt: (">", "<="),
       ast.GtE: (">=", "<"), ast.Is: (r"\bis\b", "is not"), ast.IsNot: (r"\bis\s+not\b", "is"),
       ast.In: (r"\bin\b", "not in"), ast.NotIn: (r"\bnot\s+in\b", "in")}
MAX_RUN_TESTS, RUN_TIMEOUT, MAX_HIDE, MAX_HIDE_FILES = 40, 240, 5, 3
MIRROR_MAX_LINES, MIRROR_MAX_COMMITS = 150, 4000


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ---------------------------------------------------------------- edits: {rel: [(start, end, replacement)]} on the base text
def apply_edits(src, edits):
    for a, b, rep in sorted(edits, reverse=True):
        src = src[:a] + rep + src[b:]
    return src


def disjoint(edits):
    e = sorted((a, b) for a, b, _ in edits)
    return all(e[i][1] <= e[i + 1][0] for i in range(len(e) - 1))


# ---------------------------------------------------------------- procedural mutations
def _offsets(src):
    lines = src.splitlines(keepends=True)
    starts, s = [], 0
    for l in lines:
        starts.append(s); s += len(l)
    def off(lineno, col_bytes):
        return starts[lineno - 1] + len(lines[lineno - 1].encode()[:col_bytes].decode(errors="ignore"))
    return lines, starts, off


def _owns_lines(lines, node):
    """The statement starts its first line and ends its last one (no `a = 1; b = 2`)."""
    before = lines[node.lineno - 1].encode()[:node.col_offset].decode(errors="ignore")
    after = lines[node.end_lineno - 1].encode()[node.end_col_offset:].decode(errors="ignore").strip()
    return not before.strip() and (not after or after.startswith("#"))


def _is_str(n):
    return isinstance(n, ast.JoinedStr) or (isinstance(n, ast.Constant) and isinstance(n.value, (str, bytes)))


def mutations(src):
    """Yield (op, lineno, end_lineno, function_name, edit) for every mutation site; edit = (start, end, replacement)."""
    tree = ast.parse(src)
    lines, starts, off = _offsets(src)
    seg = lambda n: (off(n.lineno, n.col_offset), off(n.end_lineno, n.end_col_offset))
    whole = lambda n0, n1: (starts[n0.lineno - 1], starts[n1.end_lineno - 1] + len(lines[n1.end_lineno - 1]))

    def between(n1, n2, pat, rep):
        a, b = off(n1.end_lineno, n1.end_col_offset), off(n2.lineno, n2.col_offset)
        mid, k = re.subn(pat, rep, src[a:b], count=1)
        return (a, b, mid) if k else None

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
                    a, b = whole(st, st)
                    if isinstance(st, ast.If) and not st.orelse:
                        yield "drop_if", st, (a, b, "")
                    elif isinstance(st, (ast.For, ast.While)) and not st.orelse:
                        yield "drop_loop", st, (a, b, "")
                    elif (isinstance(st, (ast.Assign, ast.AugAssign)) and rebound(st)) or \
                            (isinstance(st, ast.Expr) and isinstance(st.value, ast.Call)):
                        yield "drop_stmt", st, (a, b, "")
        for n in ast.walk(fn):
            if isinstance(n, ast.Compare) and len(n.ops) == 1 and type(n.ops[0]) in CMP:
                pat, rep = CMP[type(n.ops[0])]
                e = between(n.left, n.comparators[0], pat if pat.startswith("\\") else re.escape(pat), rep)
                if e: yield "flip_compare", n, e
            elif isinstance(n, ast.BoolOp) and len(n.values) == 2:
                old, new = ("and", "or") if isinstance(n.op, ast.And) else ("or", "and")
                e = between(n.values[0], n.values[1], rf"\b{old}\b", new)
                if e: yield "flip_boolop", n, e
            elif isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub)) and not (_is_str(n.left) or _is_str(n.right)):
                e = between(n.left, n.right, r"\+" if isinstance(n.op, ast.Add) else "-", "-" if isinstance(n.op, ast.Add) else "+")
                if e: yield "flip_binop", n, e
            if isinstance(n, ast.If):
                a, b = seg(n.test)
                if isinstance(n.test, ast.UnaryOp) and isinstance(n.test.op, ast.Not):
                    oa, ob = seg(n.test.operand)
                    yield "negate_if", n.test, (a, b, src[oa:ob])
                else:
                    yield "negate_if", n.test, (a, b, f"not ({src[a:b]})")
                o = n.orelse
                if o and not (len(o) == 1 and isinstance(o[0], ast.If)) and all(_owns_lines(lines, s) for s in n.body + o) \
                        and n.body[0].col_offset == o[0].col_offset:
                    ba, bb = whole(n.body[0], n.body[-1]); oa, ob = whole(o[0], o[-1])
                    yield "swap_if_else", n, (ba, ob, src[oa:ob] + src[bb:oa] + src[ba:bb])
            if isinstance(n, (ast.Compare, ast.BinOp, ast.Subscript, ast.Slice)):
                for c in ast.iter_child_nodes(n):
                    if isinstance(c, ast.Constant) and type(c.value) is int and c.value < 1000:
                        a, b = seg(c)
                        yield "const_plus_one", c, (a, b, str(c.value + 1))
            if isinstance(n, ast.keyword) and isinstance(n.value, ast.Constant) and type(n.value.value) is bool:
                a, b = seg(n.value)
                yield "flip_bool", n.value, (a, b, str(not n.value.value))
            if isinstance(n, ast.Return) and isinstance(n.value, ast.Constant) and type(n.value.value) is bool:
                a, b = seg(n.value)
                yield "flip_bool", n.value, (a, b, str(not n.value.value))

    seen = set()
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for op, node, edit in visit(fn):
                new = apply_edits(src, [edit])
                if (op, edit) in seen or new == src:
                    continue
                seen.add((op, edit))
                try:
                    ast.parse(new)
                except SyntaxError:
                    continue
                yield op, node.lineno, node.end_lineno, fn.name, edit


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
    return "".join(l for i, l in enumerate(lines, 1) if not any(a <= i <= b for a, b in drop))


def touched_functions(patches):
    """{file: {function or class names}} touched by patches (hunk headers + changed def lines)."""
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


def function_spans(src):
    return [(n.name, n.lineno, n.end_lineno) for n in ast.walk(ast.parse(src)) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


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
    _W.update(dir=d, a=argparse.Namespace(**a_dict))


def _wrun(job):
    cid, files, tests = job
    d, a = _W["dir"], _W["a"]
    old = {rel: (d / rel).read_text() for rel in files}
    try:
        for rel, txt in files.items():
            (d / rel).write_text(txt)
        x = d / f".junit_{cid}.xml"
        subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o", "junit_family=xunit1",
                        f"--junitxml={x}", *tests], cwd=d, env=env_for(d, a), capture_output=True, timeout=RUN_TIMEOUT)
        res = parse_junit(x) if x.exists() else None
        x.unlink(missing_ok=True)
        return cid, res
    except subprocess.TimeoutExpired:
        return cid, None
    finally:
        for rel, txt in old.items():
            (d / rel).write_text(txt)


def git(d, *args, env=None, check=True, inp=None):
    return subprocess.run(["git", "-C", str(d), *args], capture_output=True, text=True, check=check, env=env, input=inp).stdout


# ---------------------------------------------------------------- per repo
class Repo:
    def __init__(self, short, comp_rows, guard_ids, excluded_prs, a, rng):
        self.short, self.a, self.rng = short, a, rng
        rows = sorted([r for r in comp_rows if r["instance_id"].split("_")[0] == short], key=lambda r: r["created_at"])
        self.src_task = rows[-1]
        self.guard = touched_functions([r["patch"] for r in rows if r["instance_id"] in guard_ids])
        self.excluded_prs = excluded_prs
        self.work = Path(a.work) / short
        shutil.rmtree(self.work, ignore_errors=True)
        self.base = self.work / "base"
        self.base.mkdir(parents=True)
        with tarfile.open(Path(a.comp) / "snapshots" / f"{self.src_task['instance_id']}.tgz") as tf:
            tf.extractall(self.base, filter="tar")
        self.pkg = next(p for p in (self.base / "src" / PKG[short], self.base / PKG[short]) if p.is_dir())
        self.pkg_rel = str(self.pkg.relative_to(self.base))
        self.text = {}
        log(f"{short}: base = {self.src_task['instance_id']} @ {self.src_task['base_commit'][:10]}, package {self.pkg_rel}; "
            f"guarded functions {sum(map(len, self.guard.values()))} in {len(self.guard)} files")

    def src(self, rel):
        if rel not in self.text:
            self.text[rel] = (self.base / rel).read_text()
        return self.text[rel]

    def files_of(self, edits):
        return {rel: apply_edits(self.src(rel), e) for rel, e in edits.items()}

    def baseline(self):
        a, base, work = self.a, self.base, self.work
        rc = work / "covrc"
        rc.write_text(f"[run]\ndynamic_context = test_function\nsource = {self.pkg}\ndata_file = {work}/.coverage\n")
        t0, bj = time.time(), work / "baseline.xml"
        r = subprocess.run([sys.executable, "-m", "coverage", "run", f"--rcfile={rc}", "-m", "pytest", "-q", "-p", "no:cacheprovider",
                            "-o", "junit_family=xunit1", f"--junitxml={bj}", "--continue-on-collection-errors", "tests"],
                           cwd=base, env=env_for(base, a), capture_output=True, text=True, timeout=3600)
        if not bj.exists():
            raise RuntimeError(f"baseline produced no junit (rc {r.returncode})\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
        subprocess.run([sys.executable, "-m", "coverage", "json", f"--rcfile={rc}", "--show-contexts", "-o", str(work / "cov.json")],
                       cwd=base, env=env_for(base, a), check=True, capture_output=True)
        self.bl = parse_junit(bj)
        bad_files = {k[0] for k, (st, _) in self.bl.items() if st in ("fail", "error")}
        self.eligible = {k for k, (st, _) in self.bl.items() if st == "pass" and k[0] not in bad_files}
        log(f"{self.short}: baseline {len(self.bl)} test functions in {time.time() - t0:.0f}s; "
            f"{len(re.findall(r'ERROR collecting', r.stdout))} test files failed to collect; {len(bad_files)} test files with failures "
            f"(excluded); {len(self.eligible)} eligible F2P tests")
        by_dotted = {".".join(([f[:-3].replace("/", ".")] if f else []) + ([c] if c else []) + [n]): (f, c, n) for f, c, n in self.bl}
        memo = {}
        def ctx_key(c):
            if c not in memo:
                s = c.split("|")[0]
                hits = [by_dotted[s]] if s in by_dotted else \
                    [k for d, k in by_dotted.items() if d.endswith("." + s) or s.endswith("." + d)]
                memo[c] = hits[0] if len(hits) == 1 else None
            return memo[c]
        self.cover = {}      # rel -> {line: set(test keys)} (eligible tests only)
        for fpath, info in json.load(open(work / "cov.json"))["files"].items():
            rel = str(Path(fpath).resolve().relative_to(base.resolve())) if Path(fpath).is_absolute() else fpath
            self.cover[rel] = {int(ln): {k for k in map(ctx_key, cs) if k in self.eligible}
                               for ln, cs in info.get("contexts", {}).items()}

    def tests_for_lines(self, rel, l0, l1):
        cov = self.cover.get(rel, {})
        return set().union(*[cov.get(ln, set()) for ln in range(l0, l1 + 1)]) if cov else set()

    def evaluate(self, cands, want, label):
        """Run candidates (dicts with 'edits', 'tests') in parallel; return accepted ones (with 'res'), in order of finishing,
        stopping at `want` (accept(c) decides diversity)."""
        jobs = []
        for i, c in enumerate(cands):
            run = sorted(c["tests"])
            if len(run) > MAX_RUN_TESTS:
                run = self.rng.sample(run, MAX_RUN_TESTS)
            jobs.append((i, self.files_of(c["edits"]), [nodeid(t) for t in run]))
        q = Manager().Queue()
        for w in range(self.a.workers):
            q.put(w)
        kept, stats = [], defaultdict(int)
        with Pool(self.a.workers, initializer=_winit, initargs=(str(self.base), str(self.work), vars(self.a), q)) as pool:
            for cid, res in pool.imap_unordered(_wrun, jobs):
                c = cands[cid]
                if res is None:
                    stats["timeout_or_no_junit"] += 1; continue
                fails = {k for k, (st, _) in res.items() if st == "fail" and k in self.eligible}
                if not fails:
                    stats["no_eligible_test_fails"] += 1; continue
                c["res"], c["fails"] = res, fails
                why = c.get("accept", lambda c: None)(c)
                if why:
                    stats[why] += 1; continue
                kept.append(c); stats["kept"] += 1
                if len(kept) >= want:
                    pool.terminate(); break
        log(f"{self.short}: {label}: tried {sum(stats.values())} of {len(cands)} -> {dict(stats)}")
        return kept

    def choose_hidden(self, c, groups):
        """Hidden tests: first one failing test per bug part (groups = list of failing-test sets), then fill up to MAX_HIDE,
        over at most MAX_HIDE_FILES files. Returns sorted keys or None if they cannot be removed cleanly."""
        hide, files = [], []
        def add(k):
            if k in hide or len(hide) >= MAX_HIDE:
                return
            if k[0] not in files:
                if len(files) >= MAX_HIDE_FILES:
                    return
                files.append(k[0])
            hide.append(k)
        for g in groups:
            g = sorted(g & c["fails"])
            if g and not (set(g) & set(hide)):
                add(g[0])
        if len(hide) < len([g for g in groups if g & c["fails"]]):
            return None                      # some bug part has no failing test of its own left
        for k in sorted(c["fails"], key=lambda k: (k[0] not in files, k)):
            add(k)
        for f in files:
            if remove_test_functions(self.src(f), {(k[1], k[2]) for k in hide if k[0] == f}) is None:
                return None
        return sorted(hide)

    # ---- 2. single procedural bugs
    def singles(self, want):
        by_op, n_guard = defaultdict(list), 0
        for rel in sorted(self.cover):
            try:
                muts = list(mutations(self.src(rel)))
            except Exception:
                continue
            for op, l0, l1, fn, edit in muts:
                if fn in self.guard.get(rel, ()):
                    n_guard += 1; continue
                tests = self.tests_for_lines(rel, l0, l1)
                if tests:
                    by_op[op].append(dict(op=op, rel=rel, function=fn, line=l0, edits={rel: [edit]}, tests=tests))
        for v in by_op.values():
            self.rng.shuffle(v)
        order = [c for grp in itertools.zip_longest(*by_op.values()) for c in grp if c]   # round-robin over operators
        log(f"{self.short}: {len(order)} covered mutation sites ({', '.join(f'{k} {len(v)}' for k, v in sorted(by_op.items()))}); "
            f"{n_guard} skipped by the dev/holdout guard")
        used_fn, per_file = set(), defaultdict(int)
        def accept(c):
            if (c["rel"], c["function"]) in used_fn: return "same_function"
            if per_file[c["rel"]] >= self.a.per_file: return "file_cap"
            h = self.choose_hidden(c, [c["fails"]])
            if not h: return "unhideable_tests"
            c["hide"] = h; used_fn.add((c["rel"], c["function"])); per_file[c["rel"]] += 1
        for c in order:
            c["accept"] = accept
        return self.evaluate(order[: want * self.a.max_tries], want, "single bugs")

    # ---- 3. combined bugs
    def combos(self, singles):
        cands = []
        def combo(parts, kind):
            edits = defaultdict(list)
            for p in parts:
                for rel, e in p["edits"].items():
                    edits[rel] += e
            if not all(disjoint(e) for e in edits.values()):
                return None
            files = self.files_of(edits)
            try:
                [ast.parse(t) for t in files.values()]
            except SyntaxError:
                return None
            return dict(op=kind, rel=",".join(sorted(edits)), function=",".join(p["function"] for p in parts), line=parts[0]["line"],
                        edits=dict(edits), tests=set().union(*[p["tests"] for p in parts]), parts=parts)
        by_file, by_mod = defaultdict(list), defaultdict(list)
        for s in singles:
            by_file[s["rel"]].append(s); by_mod[str(Path(s["rel"]).parent)].append(s)
        for rel, ss in by_file.items():                       # same file: k in 2..4, <= 3 per file, each bug once
            pool_ = ss[:]; self.rng.shuffle(pool_); made = 0
            while made < 3 and len(pool_) >= 2:
                k = min(len(pool_), self.rng.choice([2, 2, 3, 4]))
                c = combo(pool_[:k], "combine_file")
                pool_ = pool_[k:]
                if c: cands.append(c); made += 1
        for mod, ss in by_mod.items():                        # same module, different files: k in 2..3, <= 6 per module
            pool_ = ss[:]; self.rng.shuffle(pool_); made = 0
            while made < 6 and len(pool_) >= 2:
                first, rest = pool_[0], pool_[1:]
                others = [s for s in rest if s["rel"] != first["rel"]]
                if not others: break
                k = self.rng.choice([2, 3])
                pick = [first] + [s for s in others if s["rel"] not in {first["rel"]}][: k - 1]
                pick = [p for i, p in enumerate(pick) if p["rel"] not in {q["rel"] for q in pick[:i]}]
                if len(pick) < 2: pool_ = rest; continue
                pool_ = [s for s in pool_ if s not in pick]
                c = combo(pick, "combine_module")
                if c: cands.append(c); made += 1
        def accept(c):
            h = self.choose_hidden(c, [p["fails"] for p in c["parts"]])
            if not h: return "a_part_lost_its_failing_test_or_unhideable"
            c["hide"] = h
        for c in cands:
            c["accept"] = accept
        log(f"{self.short}: {len(cands)} combined-bug candidates "
            f"({sum(c['op'] == 'combine_file' for c in cands)} same file, {sum(c['op'] == 'combine_module' for c in cands)} same module)")
        return self.evaluate(cands, len(cands), "combined bugs") if cands else []

    # ---- 4. PR mirror
    def mirrors(self, want):
        base, pkg = self.base, self.pkg_rel
        log_ = git(base, "log", "--no-merges", f"-n{MIRROR_MAX_COMMITS}", "--format=%H%x09%s", "HEAD", "--", pkg).splitlines()
        cands, stats = [], defaultdict(int)
        tmp = Path(tempfile.mkdtemp(dir=self.work))
        for line in log_:
            sha, subj = line.split("\t", 1)
            prs = set(re.findall(r"#(\d+)", subj))
            if prs & self.excluded_prs:
                stats["competition_or_imported_pr"] += 1; continue
            diff = git(base, "diff", f"{sha}^", sha, "--", pkg, check=False)
            n = sum(1 for l in diff.splitlines() if l[:1] in "+-" and not l.startswith(("+++", "---")))
            if not diff or n > MIRROR_MAX_LINES:
                stats["empty_or_too_big"] += 1; continue
            if "/dev/null" in diff or "\nrename " in diff:
                stats["adds_deletes_or_renames_files"] += 1; continue
            rels = re.findall(r"^\+\+\+ b/(\S+)", diff, re.M) + re.findall(r"^--- a/(\S+)", diff, re.M)
            rels = sorted({r for r in rels if r.endswith(".py")})
            if not rels or not all((base / r).exists() for r in rels):
                stats["file_missing_at_head"] += 1; continue
            if any(fn in self.guard.get(rel, ()) for rel, fns in touched_functions([diff]).items() for fn in fns):
                stats["touches_guarded_function"] += 1; continue
            shutil.rmtree(tmp); tmp.mkdir()
            for r in rels:
                (tmp / r).parent.mkdir(parents=True, exist_ok=True); shutil.copy(base / r, tmp / r)
            pr = subprocess.run(["git", "apply", "-R", "-"], cwd=tmp, input=diff, text=True, capture_output=True)
            if pr.returncode:
                stats["does_not_revert_cleanly"] += 1; continue
            new = {r: (tmp / r).read_text() for r in rels}
            if all(new[r] == self.src(r) for r in rels):
                stats["no_change"] += 1; continue
            try:
                [ast.parse(t) for t in new.values()]
            except SyntaxError:
                stats["syntax"] += 1; continue
            tests = set()
            for r in rels:                                    # tests covering the functions the commit changed
                fns = touched_functions([diff]).get(r, set())
                for name, l0, l1 in function_spans(self.src(r)):
                    if name in fns:
                        tests |= self.tests_for_lines(r, l0, l1)
            for tf in git(base, "diff", "--name-only", f"{sha}^", sha, "--", "tests").split():
                tests |= {k for k in self.eligible if k[0] == tf}  # the commit's own test files
            if not tests:
                stats["no_covering_tests"] += 1; continue
            cands.append(dict(op="pr_mirror", rel=",".join(rels), function=sha[:10], line=0, subject=subj,
                              edits={r: [(0, len(self.src(r)), new[r])] for r in rels}, tests=tests))
        shutil.rmtree(tmp, ignore_errors=True)
        log(f"{self.short}: PR mirror: {len(log_)} commits touching {pkg} -> {len(cands)} candidates; skipped {dict(stats)}")
        def accept(c):
            h = self.choose_hidden(c, [c["fails"]])
            if not h: return "unhideable_tests"
            c["hide"] = h
        for c in cands:
            c["accept"] = accept
        return self.evaluate(cands, want, "PR mirror") if cands else []

    # ---- materialize
    def write(self, tasks, out, rows_out, meta_out, prefix):
        base = self.base
        head = git(base, "log", "-1", "--format=%an%n%ae%n%aI%n%B").split("\n", 3)
        genv = dict(os.environ, GIT_AUTHOR_NAME=head[0], GIT_AUTHOR_EMAIL=head[1], GIT_AUTHOR_DATE=head[2],
                    GIT_COMMITTER_NAME=head[0], GIT_COMMITTER_EMAIL=head[1], GIT_COMMITTER_DATE=head[2])
        for j, c in enumerate(tasks):
            iid = f"{self.short}_{prefix}{j:03d}"
            snap = self.work / "snap"
            shutil.rmtree(snap, ignore_errors=True)
            shutil.copytree(base, snap, ignore=shutil.ignore_patterns(".git", "__pycache__"), symlinks=True)
            new = self.files_of(c["edits"])
            tfiles = sorted({k[0] for k in c["hide"]})
            for rel, txt in new.items():
                (snap / rel).write_text(txt)
            for tf in tfiles:
                (snap / tf).write_text(remove_test_functions(self.src(tf), {(k[1], k[2]) for k in c["hide"] if k[0] == tf}))
            git(snap, "init", "-q"); git(snap, "add", "-A")
            git(snap, "commit", "-q", "-m", head[3].strip() or "update", env=genv)
            for rel in new:
                (snap / rel).write_text(self.src(rel))
            patch = git(snap, "diff", "--", *new)
            git(snap, "checkout", "-q", "--", *new)
            for tf in tfiles:
                (snap / tf).write_text(self.src(tf))
            tpatch = git(snap, "diff", "--", *tfiles)
            git(snap, "checkout", "-q", "--", *tfiles)
            with tarfile.open(out / "snapshots" / f"{iid}.tgz", "w:gz") as tf_:
                tf_.add(snap, arcname=".")
            shutil.rmtree(snap)
            rows_out.append({"instance_id": iid, "repo": self.src_task["repo"], "base_commit": self.src_task["base_commit"],
                             "problem_statement": "", "hints_text": "", "patch": patch, "test_patch": tpatch,
                             "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ")})
            meta_out.append({"instance_id": iid, "source_task": self.src_task["instance_id"], "op": c["op"], "file": c["rel"],
                             "function": c["function"], "line": c["line"], "subject": c.get("subject", ""),
                             "n_bugs": len(c.get("parts", [c])), "parts": [p["op"] for p in c.get("parts", [])],
                             "hidden_tests": [nodeid(k) for k in c["hide"]], "n_tests_run": len(c["res"]),
                             "n_tests_failed": sum(st == "fail" for st, _ in c["res"].values()),
                             "fail_msgs": {nodeid(k): c["res"][k][1] for k in c["hide"]}})
        return len(tasks)


def cmd_build(a):
    out = Path(a.out)
    for d in ("snapshots", "graphs", "embeddings"):
        (out / d).mkdir(parents=True, exist_ok=True)
    comp_rows = [json.loads(l) for l in open(Path(a.comp) / "tasks.jsonl") if l.strip()]
    guard_ids = set(json.load(open(a.guard))["guard_task_ids"])
    excluded_prs = {r["instance_id"].rsplit("_", 1)[1] for r in comp_rows}
    for ext in a.exclude_tasks or []:
        if Path(ext).exists():
            excluded_prs |= {json.loads(l)["instance_id"].rsplit("_", 1)[1] for l in open(ext) if l.strip()}
    rng = random.Random(a.seed)
    rows, meta, counts = [], [], {}
    for short, want in a.target.items():
        t0 = time.time()
        try:
            r = Repo(short, comp_rows, guard_ids, excluded_prs, a, rng)
            r.baseline()
            s = r.singles(want)
            c = r.combos([x for x in s])
            m = r.mirrors(a.mirror_target.get(short, want))
            free = shutil.disk_usage(out).free / 2**30
            if free < a.min_free_gb:
                raise RuntimeError(f"only {free:.1f} GB free")
            counts[short] = dict(single=r.write(s, out, rows, meta, "syn"), combined=r.write(c, out, rows, meta, "cmb"),
                                 mirror=r.write(m, out, rows, meta, "pr"))
            for d, ext in (("graphs", "json"), ("embeddings", "npz")):
                g = Path(a.comp) / d / f"{short}_{r.src_task['base_commit']}.{ext}"
                shutil.copy(g, out / d / g.name) if g.exists() else log(f"{short}: WARNING no competition {d} file {g.name}")
            shutil.rmtree(r.work, ignore_errors=True)
            log(f"{short}: {counts[short]} tasks written in {(time.time() - t0) / 60:.0f} min")
        except Exception as e:
            log(f"{short}: FAILED {type(e).__name__}: {e}")
        with open(out / "tasks.jsonl", "w") as f:            # rewritten after every repo: a later failure keeps earlier repos
            f.writelines(json.dumps(r_) + "\n" for r_ in rows)
        with open(out / "synthetic_meta.jsonl", "w") as f:
            f.writelines(json.dumps(m_) + "\n" for m_ in meta)
    (out / "graphcomp_external.json").write_text(json.dumps({
        "built": [r_["instance_id"] for r_ in rows], "valid": [], "kind": "synthetic: procedural, combined, PR mirror (make_synthetic.py)",
        "counts": counts, "seed": a.seed, "target": a.target, "guard": a.guard,
        "note": "valid is filled by import_external.py --record-validity, then narrowed by the issues step"}, indent=1))
    log(f"built {len(rows)} synthetic tasks into {out}: {counts}")
    if not rows:
        sys.exit("no tasks built")


# ---------------------------------------------------------------- issues
PROMPT = """You are writing a GitHub issue for the {repo} project. A bug exists in the current code. Below is the fix the maintainers will eventually apply (do NOT reveal it) and the output of the project's tests that fail because of the bug.

Write the issue as a user who hit the bug would: a short title line, what they did, what they expected, and what happened instead, with a small reproduction script that uses the public API (it may be approximate). {multi}Do not mention the fix, line numbers, the diff, test files or test function names, and do not quote the changed lines. Output only the issue text.

FIX (hidden from the reader):
{patch}

FAILING TEST OUTPUT:
{fails}
"""
MULTI = "The fix touches several places; describe every wrong behaviour the failing tests show, as one report. "


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
    llm = LLM(model=a.model, max_model_len=12288, gpu_memory_utilization=0.88, seed=a.seed)
    sp = SamplingParams(n=4, temperature=0.8, top_p=0.95, max_tokens=1000, seed=a.seed)
    msgs = [[{"role": "user", "content": PROMPT.format(
        repo=r["repo"], patch=r["patch"][:6000], multi=MULTI if meta[r["instance_id"]]["n_bugs"] > 1 or meta[r["instance_id"]]["op"] == "pr_mirror" else "",
        fails="\n\n".join(f"{k}:\n{v[-1500:]}" for k, v in meta[r["instance_id"]]["fail_msgs"].items())[:6000])}] for r in todo]
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
        if p.name[:-4] not in keep_ids:
            p.unlink()
    meta_g.update(valid=sorted(keep_ids), issue_model=a.model, issue_rejected=rejected)
    (out / "graphcomp_external.json").write_text(json.dumps(meta_g, indent=1))
    log(f"issues: kept {len(kept)}, rejected {len(rejected)} (no sample passed the leak filter)")
    shown = set()
    for r in kept:
        op = meta[r["instance_id"]]["op"]
        if op not in shown:
            shown.add(op)
            print(f"---- {r['instance_id']} ({op})\n{r['problem_statement']}\n", flush=True)


# ---------------------------------------------------------------- self-test
def selftest():
    src = ("def f(x, y):\n    z = x + 1\n    if x == y and y > 2:\n        z = 0\n    print(z)\n    for i in y:\n        z += i\n"
           "    if z:\n        z = 1\n    else:\n        z = 2\n    g(flag=True)\n    return z if x[1:3] else None\n\n"
           "class C:\n    def g(self, a):\n        a.b = 3; return a\n")
    got = list(mutations(src))
    ops = {o for o, *_ in got}
    assert {"flip_compare", "flip_boolop", "negate_if", "drop_if", "drop_stmt", "const_plus_one", "flip_binop", "drop_loop",
            "swap_if_else", "flip_bool"} <= ops, ops
    assert all(fn == "f" for *_, fn, _ in got), "g's statements share a line"
    news = {o: [] for o in ops}
    for op, *_, e in got:
        n = apply_edits(src, [e]); ast.parse(n); assert n != src; news[op].append(n)
    assert any("x != y" in n for n in news["flip_compare"]) and any("y <= 2" in n for n in news["flip_compare"])
    assert any("x == y or y > 2" in n for n in news["flip_boolop"])
    assert any("if not (x == y and y > 2):" in n for n in news["negate_if"])
    assert any("z = x - 1" in n for n in news["flip_binop"])
    assert any("if z:\n        z = 2\n    else:\n        z = 1\n" in n for n in news["swap_if_else"]), news["swap_if_else"]
    assert any("flag=False" in n for n in news["flip_bool"])
    e1 = [e for o, *_, e in got if o == "flip_compare"][0]; e2 = [e for o, *_, e in got if o == "flip_bool"][0]
    assert disjoint([e1, e2]) and "flag=False" in apply_edits(src, [e1, e2]) and not disjoint([e1, e1])
    ne = "def h(a):\n    m = a + 1\n    print(m)\n    k = 0\n    k = a\n    return k\n"
    dropped = [apply_edits(ne, [e]) for o, *_, e in mutations(ne) if o == "drop_stmt"]
    assert all("m = a + 1" in n for n in dropped) and any(n.count("k =") == 1 for n in dropped), dropped
    t = "import pytest\n\n@pytest.mark.x\ndef test_a():\n    assert 1\n\ndef test_b():\n    pass\n\nclass TestK:\n    def test_c(self):\n        pass\n"
    r = remove_test_functions(t, {("", "test_a")})
    assert "test_a" not in r and "@pytest.mark.x" not in r and "test_b" in r
    assert remove_test_functions(t, {("TestK", "test_c")}) is None and remove_test_functions(t, {("", "test_zz")}) is None
    assert touched_functions(["+++ b/pkg/m.py\n@@ -1,2 +1,2 @@ def foo(a):\n-    x\n+    y\n+def bar():\n"]) == {"pkg/m.py": {"foo", "bar"}}
    row = {"patch": "--- a/m.py\n+++ b/m.py\n-    if left != right:\n+    if left == right:\n"}
    m = {"hidden_tests": ["tests/t.py::test_eq"]}
    assert not issue_ok("x" * 300 + "test_eq", row, m) and not issue_ok("x" * 300 + "if left == right:", row, m)
    assert issue_ok("Calling f with equal values returns the wrong branch. " * 6, row, m)
    x = ET.fromstring('<testsuite><testcase file="t.py" classname="t" name="test_a[1]"><error message="port"/></testcase>'
                      '<testcase file="t.py" classname="t" name="test_a[2]"/><testcase file="t.py" classname="t.K" name="test_b">'
                      '<failure message="assert"/></testcase></testsuite>')
    Path("/tmp/_junit_selftest.xml").write_bytes(ET.tostring(x))
    j = parse_junit("/tmp/_junit_selftest.xml")
    assert j[("t.py", "", "test_a")][0] == "error" and j[("t.py", "K", "test_b")][0] == "fail", j
    # choose_hidden: one failing test per bug part first
    class R: pass
    rr = R(); rr.src = lambda f: "def test_1():\n    pass\n\ndef test_2():\n    pass\n\ndef test_3():\n    pass\n"
    k1, k2, k3 = ("a.py", "", "test_1"), ("a.py", "", "test_2"), ("a.py", "", "test_3")
    c = {"fails": {k1, k2, k3}}
    h = Repo.choose_hidden(rr, c, [{k3}, {k1, k2}])
    assert h is not None and k3 in h and (k1 in h or k2 in h), h
    assert Repo.choose_hidden(rr, {"fails": {k1}}, [{k1}, {k1}]) is None, "two parts sharing their only failing test"
    print("selftest ok")


def main():
    if "--selftest" in sys.argv:
        return selftest()
    kv = lambda s: {k: int(v) for k, v in (p.split("=") for p in s.split(","))}
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--comp", required=True); b.add_argument("--out", required=True); b.add_argument("--work", required=True)
    b.add_argument("--tools", required=True, help="dir with `coverage` (pip --target), first on PYTHONPATH")
    b.add_argument("--guard", required=True, help="guard_tasks.json")
    b.add_argument("--exclude-tasks", nargs="*", help="tasks.jsonl files whose PR numbers are never mirrored (imported tasks)")
    b.add_argument("--target", default="fastapi=300,rich=250,requests=80,httpx=80", type=kv, help="single bugs per repo")
    b.add_argument("--mirror-target", default="fastapi=300,rich=200,requests=60,httpx=60", type=kv)
    b.add_argument("--per-file", type=int, default=8, help="max single bugs per source file")
    b.add_argument("--workers", type=int, default=8); b.add_argument("--max-tries", type=int, default=4)
    b.add_argument("--min-free-gb", type=float, default=8); b.add_argument("--seed", type=int, default=1)
    i = sub.add_parser("issues")
    i.add_argument("--out", required=True); i.add_argument("--model", required=True); i.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    cmd_build(a) if a.cmd == "build" else cmd_issues(a)


if __name__ == "__main__":
    main()
