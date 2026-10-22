"""Official E2E NLG Challenge scoring, by running tuetschek/e2e-metrics itself.

BLEU and NIST come from its Python port of mteval-v13a (``-p``), METEOR / ROUGE-L / CIDEr
from its MS-COCO scripts behind the Stanford PTB tokenizer.  The last three need Java.

Scores are returned in the convention of the LoRA paper's E2E table:
BLEU, METEOR and ROUGE-L x100, NIST and CIDEr as reported.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

REPO_URL = "https://github.com/tuetschek/e2e-metrics"
SETUP_HINT = "run setup_metrics.sh, or point --e2e-metrics / $E2E_METRICS at a clone of " + REPO_URL
SCALE = {"BLEU": 100.0, "NIST": 1.0, "METEOR": 100.0, "ROUGE_L": 100.0, "CIDEr": 1.0}
KEYS = {"BLEU": "bleu", "NIST": "nist", "METEOR": "meteor", "ROUGE_L": "rouge_l", "CIDEr": "cider"}


def find(path=None):
    cands = [path, os.environ.get("E2E_METRICS"),
             os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "e2e-metrics")]
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "measure_scores.py")):
            return os.path.abspath(c)
    raise FileNotFoundError(f"e2e-metrics not found; {SETUP_HINT}")


def java_env():
    """os.environ with java on PATH, taking $JAVA_HOME/bin when PATH lacks it.

    pycocoevalcap spawns a bare ``java`` for the PTB tokenizer and METEOR, so the
    child process must find it through PATH.
    """
    env = os.environ.copy()
    if shutil.which("java") is None:
        home = env.get("JAVA_HOME")
        jbin = os.path.join(home, "bin") if home else None
        if not (jbin and shutil.which("java", path=jbin)):
            raise RuntimeError("java not found on PATH or under $JAVA_HOME; METEOR and the "
                               "PTB tokenizer need Java 1.8+ (Ubuntu: "
                               "sudo apt-get install -y default-jre-headless)")
        env["PATH"] = jbin + os.pathsep + env.get("PATH", "")
    return env


def score(refs_path, hyps_path, e2e_dir=None, timeout=3600):
    """Return {bleu, nist, meteor, rouge_l, cider} for one hyps/refs pair."""
    e2e_dir = find(e2e_dir)
    cmd = [sys.executable, os.path.join(e2e_dir, "measure_scores.py"), "-p", "-t",
           os.path.abspath(refs_path), os.path.abspath(hyps_path)]
    r = subprocess.run(cmd, cwd=e2e_dir, env=java_env(), capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=timeout)
    lines = [l for l in r.stdout.splitlines() if l.strip()]
    if r.returncode != 0 or not lines:
        raise RuntimeError(f"measure_scores.py failed (exit {r.returncode}):\n"
                           + "\n".join(r.stderr.splitlines()[-15:]))
    cells = lines[-1].split("\t")
    if len(cells) != 6:
        raise RuntimeError(f"unexpected measure_scores.py output: {lines[-1]!r}")
    return {KEYS[m]: round(float(v) * SCALE[m], 4) for m, v in zip(SCALE, cells[1:])}


def check(e2e_dir=None):
    """Score a two-sentence example end to end; raises if any piece is missing."""
    e2e_dir = find(e2e_dir)
    with tempfile.TemporaryDirectory() as d:
        refs, hyps = os.path.join(d, "refs.txt"), os.path.join(d, "hyps.txt")
        with open(refs, "w", encoding="utf-8") as f:
            f.write("The Vaults is a pub near Café Adriatic.\nThe Vaults pub is near Café Adriatic.\n\n"
                    "Blue Spice is a cheap coffee shop in the city centre.\n")
        with open(hyps, "w", encoding="utf-8") as f:
            f.write("The Vaults is a pub near Café Adriatic.\n"
                    "Blue Spice is a coffee shop in the city centre with low prices.\n")
        s = score(refs, hyps, e2e_dir)
    if not 0 < s["bleu"] <= 100:
        raise RuntimeError(f"scorer returned nonsense: {s}")
    return e2e_dir
