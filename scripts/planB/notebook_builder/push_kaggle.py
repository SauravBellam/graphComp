"""Push Plan B notebooks to Kaggle with the right inputs, image and accelerator (kaggle CLI >= 2.2).

Credentials come from the environment (KAGGLE_USERNAME / KAGGLE_KEY, or KAGGLE_CONFIG_DIR pointing at kaggle.json);
nothing is written into a notebook. NOTE: a push starts a run (the same as Save Version -> Save & Run All).

    python push_kaggle.py ref   --ref USER/working-notebook          # show the reference image / accelerator / sources
    python push_kaggle.py push  --ref USER/working-notebook 01 [--resume] [--dry-run]
    python push_kaggle.py status 01
    python push_kaggle.py output 01 --dest DIR                         # download the latest version's output
    python push_kaggle.py quota
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
NB_DIR = Path(os.environ.get("NB_DIR", HERE.parents[2] / "planB_results" / "kaggle_upload" / "notebooks"))
COMPETITION = "gemma-4-developer-agent"
DATASETS = ["metric/gemma-4-developer-agent-wheelhouse"]
MODELS = ["google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2"]
TITLES = {  # notebook number -> (file, Kaggle title); the slug is the slugified title
    "01": ("01_validity_B2_s1.ipynb", "graphcomp 01 validity B2 s1"),
    "02": ("02_B3_s1.ipynb", "graphcomp 02 B3 s1"),
    "03": ("03_forced_call_dev_s1.ipynb", "graphcomp 03 forced call dev s1"),
    "04": ("04_qualified_names_dev_s1.ipynb", "graphcomp 04 qualified names dev s1"),
    "05": ("05_B2_s2.ipynb", "graphcomp 05 B2 s2"),
    "06": ("06_B3_s2.ipynb", "graphcomp 06 B3 s2"),
    "07": ("07_B2_s3.ipynb", "graphcomp 07 B2 s3"),
    "08": ("08_B3_s3.ipynb", "graphcomp 08 B3 s3"),
    "09": ("09_thinking0_dev_s1.ipynb", "graphcomp 09 thinking0 dev s1"),
    "10": ("10_outreserve8k_dev_s1.ipynb", "graphcomp 10 outreserve8k dev s1"),
}


def slugify(t):
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", t.lower())).strip("-")


def user():
    u = os.environ.get("KAGGLE_USERNAME")
    if not u:
        cfg = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".config" / "kaggle")) / "kaggle.json"
        u = json.load(open(cfg))["username"]
    return u


def kid(n):
    return f"{user()}/{slugify(TITLES[n][1])}"


def run(cmd, check=True):
    print("$", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    print((r.stdout + r.stderr).strip(), flush=True)
    if check and r.returncode:
        sys.exit(r.returncode)
    return r


def ref_meta(ref):
    d = tempfile.mkdtemp()
    run(["kaggle", "kernels", "pull", ref, "-p", d, "-m"])
    m = json.load(open(Path(d) / "kernel-metadata.json"))
    shutil.rmtree(d)
    return m


def push(n, ref, resume=False, dry=False):
    m = ref_meta(ref)
    if not m.get("docker_image") and not m.get("machine_shape"):
        print("WARNING: the reference notebook's metadata has no docker_image / machine_shape:", m)
    srcs = [] if n == "01" else [kid("01")]
    if resume:
        srcs.append(kid(n))
    meta = {
        "id": kid(n), "title": TITLES[n][1], "code_file": TITLES[n][0], "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_internet": False,
        "machine_shape": m.get("machine_shape"), "docker_image": m.get("docker_image"),
        "competition_sources": sorted(set([COMPETITION] + list(m.get("competition_sources") or []))),
        "dataset_sources": sorted(set(DATASETS + [s for s in (m.get("dataset_sources") or []) if "wheelhouse" in s])),
        "model_sources": sorted(set(MODELS)),
        "kernel_sources": srcs,
    }
    print(json.dumps(meta, indent=1))
    if dry:
        print("dry run: not pushed")
        return
    d = Path(tempfile.mkdtemp())
    shutil.copy(NB_DIR / TITLES[n][0], d / TITLES[n][0])
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    run(["kaggle", "kernels", "push", "-p", str(d)])
    shutil.rmtree(d)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["ref", "push", "status", "output", "quota"])
    ap.add_argument("n", nargs="?")
    ap.add_argument("--ref")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--dest", default="kaggle_output")
    a = ap.parse_args()
    if a.cmd == "ref":
        print(json.dumps(ref_meta(a.ref), indent=1))
    elif a.cmd == "push":
        push(a.n, a.ref, a.resume, a.dry_run)
    elif a.cmd == "status":
        run(["kaggle", "kernels", "status", kid(a.n)], check=False)
    elif a.cmd == "output":
        Path(a.dest).mkdir(parents=True, exist_ok=True)
        run(["kaggle", "kernels", "output", kid(a.n), "-p", a.dest], check=False)
    elif a.cmd == "quota":
        run(["kaggle", "quota"] if shutil.which("kaggle") else ["true"], check=False)
