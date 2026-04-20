"""
status.py
---------
One-shot status / monitoring for the Phase 1a run on GCP.

No arguments needed in the common case — it reads the orchestrator's
state file to find the VM, checks GCP for VM + GCS state, SSHes once
to pull the last few lines of phase1a.log, parses the latest progress
line, and prints a compact, color-coded status block.

Useful signals it surfaces at a glance:
  * VM state (RUNNING / TERMINATED / missing)
  * Provisioning model (spot vs on-demand)
  * Current training step + % complete + ETA
  * Mean/std reward weights (are fear + affiliation emerging?)
  * Population counts + mean energy (bug-canary for pop collapse)
  * GCS checkpoint count + most recent checkpoint age
  * Recent exits / retries from the phase1a log tail

Usage:
  source scripts/gcloud-env.sh
  python3 scripts/status.py                # one-shot
  python3 scripts/status.py --follow 30    # refresh every 30s
  python3 scripts/status.py --log 40       # include the last 40 log lines
"""

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import time


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_FILE = os.path.join(REPO_ROOT, ".claude-orch-state.json")
VM_NAME = "evo-reward-gpu"
GCS_BUCKET = "evo-reward-ckpts"

# Hardcode the project + account so this script is safe to run from any
# terminal without first sourcing scripts/gcloud-env.sh. If the user's
# global gcloud config happens to point at a different project with a
# same-named VM, that'd return bogus data otherwise.
GCP_PROJECT = "evo-reward"
GCP_ACCOUNT = "db3792@columbia.edu"

TOTAL_STEPS_DEFAULT = 10_240_000  # baseline_faithful.yaml

# ANSI color codes
C_RESET = "\033[0m"
C_DIM = "\033[2m"
C_BOLD = "\033[1m"
C_GREEN = "\033[32m"
C_YELLOW = "\033[33m"
C_RED = "\033[31m"
C_CYAN = "\033[36m"
C_MAGENTA = "\033[35m"


def color_enabled():
    return sys.stdout.isatty()


def c(text, code):
    return f"{code}{text}{C_RESET}" if color_enabled() else text


def gcloud(*args, timeout=60):
    # Run gcloud with project+account pinned via env vars, so the caller's
    # shell-global gcloud config can't redirect us to a different project.
    env = {**os.environ,
           "CLOUDSDK_CORE_PROJECT": GCP_PROJECT,
           "CLOUDSDK_CORE_ACCOUNT": GCP_ACCOUNT}
    try:
        r = subprocess.run(["gcloud"] + list(args), capture_output=True,
                           text=True, timeout=timeout, env=env)
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except subprocess.TimeoutExpired:
        return -1, "", "TIMEOUT"


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


# ─── VM-level info ─────────────────────────────────────────────────────────

def vm_info(zone):
    """Return a dict of VM info, or None if the VM doesn't exist.

    If describe returns TERMINATED on the first call, retry once after
    a short delay — we've seen gcloud very occasionally return stale
    data that flips back on the retry. Avoids confusing alerts.
    """
    if not zone:
        return None
    fmt = ("value(status,machineType.basename(),"
           "scheduling.provisioningModel,creationTimestamp,lastStartTimestamp)")

    for attempt in range(2):
        rc, out, _ = gcloud("compute", "instances", "describe", VM_NAME,
                            f"--zone={zone}", f"--format={fmt}")
        if rc != 0:
            return None
        parts = out.split("\t")
        while len(parts) < 5:
            parts.append("")
        status = parts[0]
        # Only retry on the alarming states — RUNNING / STAGING etc. we trust.
        if status == "TERMINATED" and attempt == 0:
            time.sleep(2)
            continue
        break

    return {
        "status": parts[0],
        "machine": parts[1],
        "provisioning": parts[2] or "STANDARD",
        "created_at": parts[3],
        "last_started_at": parts[4],
    }


# ─── GCS checkpoint info ───────────────────────────────────────────────────

def gcs_info():
    """Count checkpoint files in the bucket; return (count, latest_step, age_s)."""
    rc, out, _ = gcloud("storage", "ls", "--long",
                        f"gs://{GCS_BUCKET}/results/**/step_*.npz",
                        timeout=30)
    if rc != 0 or not out.strip():
        return 0, None, None
    lines = [l for l in out.strip().split("\n") if "step_" in l]
    if not lines:
        return 0, None, None

    # gcloud storage ls --long format:
    #   <size>  <ISO_timestamp>  gs://bucket/path/step_00000500.npz
    latest_step = -1
    latest_ts = None
    total = 0
    step_re = re.compile(r"step_(\d+)\.npz$")
    for line in lines:
        fields = line.split()
        if len(fields) < 3:
            continue
        total += 1
        path = fields[-1]
        m = step_re.search(path)
        if m:
            step = int(m.group(1))
            if step > latest_step:
                latest_step = step
                latest_ts = fields[-2]
    if latest_ts:
        try:
            t = dt.datetime.fromisoformat(latest_ts.replace("Z", "+00:00"))
            age = (dt.datetime.now(dt.timezone.utc) - t).total_seconds()
        except Exception:
            age = None
    else:
        age = None
    return total, latest_step if latest_step >= 0 else None, age


# ─── SSH-derived info ──────────────────────────────────────────────────────

def ssh_tail_log(zone, lines=40, timeout=30):
    """Fetch last N lines of ~/phase1a.log from the VM. Returns '' on failure."""
    rc, out, _ = gcloud(
        "compute", "ssh", VM_NAME, f"--zone={zone}", "--tunnel-through-iap",
        f"--command=tail -n {lines} ~/phase1a.log 2>/dev/null || true",
        timeout=timeout,
    )
    return out if rc == 0 else ""


def gpu_util(zone, timeout=20):
    rc, out, _ = gcloud(
        "compute", "ssh", VM_NAME, f"--zone={zone}", "--tunnel-through-iap",
        "--command=nvidia-smi --query-gpu=utilization.gpu,memory.used "
        "--format=csv,noheader,nounits 2>/dev/null || true",
        timeout=timeout,
    )
    if rc != 0 or "," not in out:
        return None, None
    util, mem = out.split(",", 1)
    try:
        return int(util.strip()), int(mem.strip())
    except ValueError:
        return None, None


# ─── Log parsing ───────────────────────────────────────────────────────────

# Matches: "Step    5000/10240000 | prey=186 pred=12 food=257 | E=131.8 |
#          w_pred=-0.04±0.47 w_prey=-0.00±0.35 | 5.1 sps | 98s"
_LOG_RE = re.compile(
    r"Step\s+(?P<step>\d+)/(?P<total>\d+)\s*\|\s*"
    r"prey=\s*(?P<prey>\d+)\s+pred=\s*(?P<pred>\d+)\s+food=\s*(?P<food>\d+)\s*\|\s*"
    r"E=\s*(?P<energy>[-+0-9.]+)\s*\|\s*"
    r"w_pred=(?P<wpd_mean>[-+0-9.]+)±(?P<wpd_std>[0-9.]+)\s+"
    r"w_prey=(?P<wpy_mean>[-+0-9.]+)±(?P<wpy_std>[0-9.]+)\s*\|\s*"
    r"(?P<sps>[0-9.]+)\s+sps\s*\|\s*(?P<elapsed>[0-9.]+)s"
)


def parse_latest_progress(log_text):
    latest = None
    for line in log_text.splitlines():
        m = _LOG_RE.search(line)
        if m:
            latest = m.groupdict()
    return latest


def count_retries(log_text):
    return sum(1 for line in log_text.splitlines() if line.startswith("[phase1a] exit"))


# ─── Rendering ─────────────────────────────────────────────────────────────

def fmt_duration(seconds):
    if seconds is None:
        return "—"
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h >= 24:
        return f"{h//24}d {h%24:02d}h"
    return f"{h:02d}:{m:02d}:{s:02d}"


def status_color(status):
    if status == "RUNNING":
        return c(status, C_GREEN)
    if status in ("TERMINATED", "STOPPED"):
        return c(status, C_YELLOW)
    return c(status, C_RED)


def render(args):
    term_width = shutil.get_terminal_size((80, 20)).columns
    bar = "═" * min(term_width, 80)

    print()
    print(c(bar, C_DIM))
    title = f" Phase 1a — {VM_NAME} @ {time.strftime('%Y-%m-%d %H:%M:%S')}"
    print(c(title, C_BOLD))
    print(c(bar, C_DIM))

    state = load_state()
    zone = state.get("zone")

    # ── VM section ────────────────────────────────────────────────────────
    info = vm_info(zone) if zone else None
    if info is None:
        print(c("\nVM   ", C_DIM), c("missing", C_RED),
              f" (state zone: {zone or 'none'})")
        print(f"\n{c('Orchestrator state:', C_DIM)} {json.dumps(state, indent=2)}")
        return

    print(f"\n{c('VM    ', C_DIM)}{status_color(info['status'])}  "
          f"{c(zone, C_CYAN)}  {info['machine']}  "
          f"{c('(' + info['provisioning'].lower() + ')', C_DIM)}")

    # ── GCS checkpoints ───────────────────────────────────────────────────
    ckpt_count, latest_step, ckpt_age = gcs_info()
    if ckpt_count:
        age_str = fmt_duration(ckpt_age) if ckpt_age is not None else "—"
        print(f"{c('GCS   ', C_DIM)}gs://{GCS_BUCKET}  "
              f"{ckpt_count} ckpt(s), latest step {latest_step:,} ({age_str} ago)")
    else:
        print(f"{c('GCS   ', C_DIM)}gs://{GCS_BUCKET}  "
              f"{c('no checkpoints yet', C_DIM)}")

    # If VM isn't RUNNING we stop here — SSH won't work.
    if info["status"] != "RUNNING":
        print(f"\n{c('VM is not running — skipping live stats.', C_DIM)}")
        print(f"{c('Last orchestrator state:', C_DIM)}")
        for k, v in state.items():
            print(f"  {k}: {v}")
        return

    # ── Live via SSH ──────────────────────────────────────────────────────
    log_text = ssh_tail_log(zone, lines=max(args.log, 40))
    progress = parse_latest_progress(log_text)
    retries = count_retries(log_text)
    util, mem = gpu_util(zone)

    if progress:
        step = int(progress["step"])
        total = int(progress["total"])
        pct = 100.0 * step / total
        sps = float(progress["sps"])
        elapsed_in_run = float(progress["elapsed"])
        remaining_steps = total - step
        eta_sec = remaining_steps / sps if sps > 0 else None

        print(f"\n{c('Training', C_BOLD)}")
        print(f"  step       {step:>10,}/{total:,}  ({pct:5.2f}%)")
        rate_line = f"  rate       {sps:>6.1f} steps/s"
        if eta_sec is not None:
            rate_line += f"   {c('ETA:', C_DIM)} {fmt_duration(eta_sec)}"
        print(rate_line)
        print(f"  elapsed    {fmt_duration(elapsed_in_run)} "
              f"{c('(current invocation)', C_DIM)}")
        if util is not None:
            gpu_txt = f"{util}% util, {mem} MiB"
            print(f"  gpu        {gpu_txt}")

        print(f"\n{c('Population', C_BOLD)}")
        print(f"  prey {progress['prey']:>4}   pred {progress['pred']:>3}   "
              f"food {progress['food']:>4}   E={float(progress['energy']):.1f}")

        # Evolving reward weights — the Phase 1a science signal
        wpd_m = float(progress["wpd_mean"])
        wpy_m = float(progress["wpy_mean"])
        fear_ok = wpd_m < 0
        soc_ok = wpy_m > 0
        fear_tag = c("← fear emerging (want <0)", C_GREEN if fear_ok else C_DIM)
        soc_tag = c("← social affiliation (want >0)", C_GREEN if soc_ok else C_DIM)

        print(f"\n{c('Reward weights (prey)', C_BOLD)}")
        print(f"  w_pred   {wpd_m:+.3f} ± {float(progress['wpd_std']):.3f}   {fear_tag}")
        print(f"  w_prey   {wpy_m:+.3f} ± {float(progress['wpy_std']):.3f}   {soc_tag}")
    else:
        print(f"\n{c('Training', C_BOLD)}")
        print(f"  {c('no progress line yet (still JIT compiling or waiting for first log)', C_DIM)}")
        if util is not None:
            print(f"  gpu        {util}% util, {mem} MiB")

    if retries:
        print(f"\n{c('Retries', C_BOLD)} {c(str(retries), C_YELLOW)} "
              f"(Python-level crashes, absorbed by phase1a_loop.sh)")

    # ── Tail of log ───────────────────────────────────────────────────────
    if args.log > 0:
        print(f"\n{c(f'Last {args.log} log lines', C_DIM)}")
        tail = "\n".join(log_text.splitlines()[-args.log:])
        print(c(tail, C_DIM))

    print(f"\n{c('watch:  gcloud compute ssh ' + VM_NAME + ' --zone=' + zone + ' --tunnel-through-iap --command=\"tail -f ~/phase1a.log\"', C_DIM)}")
    print(f"{c('attach: gcloud compute ssh ' + VM_NAME + ' --zone=' + zone + ' --tunnel-through-iap -- -t \"tmux attach -t phase1a\"', C_DIM)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--follow", type=int, nargs="?", const=30, default=None,
                    help="Refresh every N seconds (default 30 when flag present).")
    ap.add_argument("--log", type=int, default=0,
                    help="Include the last N lines of phase1a.log in output.")
    args = ap.parse_args()

    if args.follow is None:
        render(args)
    else:
        try:
            while True:
                # Clear screen (ANSI) then re-render
                if sys.stdout.isatty():
                    sys.stdout.write("\033[2J\033[H")
                render(args)
                sys.stdout.flush()
                time.sleep(args.follow)
        except KeyboardInterrupt:
            print()


if __name__ == "__main__":
    main()
