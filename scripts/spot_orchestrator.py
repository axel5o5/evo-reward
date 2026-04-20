"""
spot_orchestrator.py
--------------------
Preemption-hardened orchestrator for a spot-L4 Phase 1a run.

Architecture:
  * Provisioning happens ON the VM via a GCP startup-script (see
    scripts/vm_startup.sh), which is idempotent and runs on every boot
    including after preemption. The orchestrator does NOT install
    packages over SSH — that was the previous fragility, because spot
    VMs often get preempted mid-install.
  * The orchestrator only: (1) creates/restarts the VM, (2) uploads the
    repo tarball + phase1a retry-loop script once the base env is ready,
    (3) kicks off the tmux training session the first time. On
    preemption restart, the startup-script re-runs and re-launches
    tmux automatically — the orchestrator just has to notice the VM
    is back RUNNING.
  * Every operation is idempotent + retry-safe; the main loop never
    exits on failure, only on Ctrl-C.

Usage:
  source scripts/gcloud-env.sh
  caffeinate -d -i python3 scripts/spot_orchestrator.py --seed 0

State in .claude-orch-state.json so re-invocations resume the run.
"""

import argparse
import json
import os
import subprocess
import sys
import time


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_FILE = os.path.join(REPO_ROOT, ".claude-orch-state.json")
STARTUP_SCRIPT = os.path.join(REPO_ROOT, "scripts", "vm_startup.sh")

ZONES_TO_TRY = [
    "us-central1-a", "us-east4-b", "us-east4-c",
    "us-west1-a", "us-west1-b",
    "us-central1-b", "us-central1-c",
    "us-east1-b", "us-east1-c",
]

VM_NAME = "evo-reward-gpu"
ROUTER_PREFIX = "evo-reward-router"
NAT_PREFIX = "evo-reward-nat"
GCS_BUCKET = "evo-reward-ckpts"    # must be globally unique; change if collision

# After this many consecutive stockout-on-restart failures in the same zone,
# delete the VM and fall back to polling all zones for fresh capacity.
# Prevents getting wedged in a stuck zone with no fallback.
MAX_CONSECUTIVE_STOCKOUTS = 3

POLL_INTERVAL = 600          # 10 min between capacity polls
MONITOR_INTERVAL = 300       # 5 min between reconcile cycles
SSH_ATTEMPTS = 12            # ~7 min total
SSH_BACKOFF = 30             # seconds per attempt

# Ready-marker wait is INTENTIONALLY short (per reconcile cycle) so that
# a preempted-mid-install VM is detected quickly rather than wasted on
# a 30-min wait that's already failing. Install-in-progress across
# multiple cycles is fine — the startup-script is idempotent.
READY_MARKER_ATTEMPTS = 10   # ~5 min per reconcile cycle
READY_MARKER_BACKOFF = 30


def region_of(zone: str) -> str:
    return zone.rsplit("-", 1)[0]


def now() -> str:
    return time.strftime("[%Y-%m-%d %H:%M:%S]")


def gcloud(*args, timeout=300):
    try:
        r = subprocess.run(["gcloud"] + list(args), capture_output=True,
                           text=True, timeout=timeout)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired:
        return -1, "TIMEOUT"


def ssh(zone, cmd, timeout=120):
    return gcloud(
        "compute", "ssh", VM_NAME,
        f"--zone={zone}", "--tunnel-through-iap",
        f"--command={cmd}",
        timeout=timeout,
    )


def scp_up(zone, local_path, remote_path, timeout=600):
    return gcloud(
        "compute", "scp", local_path, f"{VM_NAME}:{remote_path}",
        f"--zone={zone}", "--tunnel-through-iap",
        timeout=timeout,
    )


# ─── Resource management ────────────────────────────────────────────────────

def ensure_nat(zone):
    region = region_of(zone)
    # Our us-central1 resources use the unsuffixed names we created originally.
    if region == "us-central1":
        router, nat = ROUTER_PREFIX, NAT_PREFIX
    else:
        router = f"{ROUTER_PREFIX}-{region.split('-')[1]}"
        nat = f"{NAT_PREFIX}-{region.split('-')[1]}"

    rc, _ = gcloud("compute", "routers", "describe", router,
                   f"--region={region}", "--format=value(name)")
    if rc == 0:
        return

    print(f"{now()} creating Cloud Router + NAT in {region}", flush=True)
    gcloud("compute", "routers", "create", router,
           f"--region={region}", "--network=default")
    gcloud("compute", "routers", "nats", "create", nat,
           f"--router={router}", f"--region={region}",
           "--nat-all-subnet-ip-ranges", "--auto-allocate-nat-external-ips")


def vm_status(zone):
    rc, out = gcloud("compute", "instances", "describe", VM_NAME,
                     f"--zone={zone}", "--format=value(status)")
    return out.strip() if rc == 0 else "UNKNOWN"


def ensure_bucket():
    """Create the GCS checkpoint bucket if it doesn't exist. Idempotent."""
    rc, _ = gcloud("storage", "buckets", "describe", f"gs://{GCS_BUCKET}",
                   "--format=value(name)")
    if rc == 0:
        return
    print(f"{now()} creating GCS bucket gs://{GCS_BUCKET}", flush=True)
    rc, out = gcloud("storage", "buckets", "create", f"gs://{GCS_BUCKET}",
                     "--location=us-central1")
    if rc != 0 and "already exists" not in out and "already own" not in out:
        print(f"{now()} bucket create failed: {out.strip()[-300:]}", flush=True)


def try_create(zone, spot=True):
    provisioning = "SPOT" if spot else "STANDARD"
    print(f"{now()} trying {zone} ({provisioning})...", flush=True)
    args = [
        "compute", "instances", "create", VM_NAME,
        f"--zone={zone}",
        "--machine-type=g2-standard-8",
        "--accelerator=type=nvidia-l4,count=1",
        "--image-family=common-cu129-ubuntu-2204-nvidia-580",
        "--image-project=deeplearning-platform-release",
        "--boot-disk-size=200GB",
        "--maintenance-policy=TERMINATE",
        f"--metadata=install-nvidia-driver=True,gcs-bucket={GCS_BUCKET}",
        f"--metadata-from-file=startup-script={STARTUP_SCRIPT}",
        "--scopes=https://www.googleapis.com/auth/cloud-platform",
        "--no-address",
    ]
    if spot:
        args += ["--provisioning-model=SPOT",
                 "--instance-termination-action=STOP"]
    rc, out = gcloud(*args)
    ok = rc == 0 and "RUNNING" in out
    if ok:
        print(f"{now()} ✅ capacity in {zone} ({provisioning})", flush=True)
        ensure_nat(zone)
    elif out:
        # Surface the actual failure reason — stockout is common, but so
        # are quota issues and other errors that need human action.
        summary = _summarize_error(out)
        if summary:
            print(f"{now()}   ↳ {summary}", flush=True)
    return ok


def _summarize_error(gcloud_output: str) -> str:
    """Pull the most useful one-line summary out of a gcloud error dump."""
    text = gcloud_output.lower()
    if "stockout" in text or "does not have enough resources" in text:
        return "stockout (zone out of capacity)"
    if "ssd_total_gb" in text or "ssd quota" in text:
        return "SSD quota exceeded — delete old VMs/disks in this region"
    if "quota" in text:
        # Try to extract the quota name
        for line in gcloud_output.splitlines():
            if "quota" in line.lower() and "exceeded" in line.lower():
                return line.strip()[:160]
        return "quota exceeded (see full gcloud output for detail)"
    if "permission" in text:
        return "permission denied"
    if "already exists" in text:
        return "VM with this name already exists"
    # Fallback: last non-empty line, truncated
    for line in reversed(gcloud_output.splitlines()):
        if line.strip():
            return line.strip()[:160]
    return ""


def poll_for_capacity(spot=True):
    while True:
        for zone in ZONES_TO_TRY:
            if try_create(zone, spot=spot):
                return zone
        mode = "spot" if spot else "on-demand"
        print(f"{now()} all zones stockout ({mode}), sleeping {POLL_INTERVAL}s",
              flush=True)
        time.sleep(POLL_INTERVAL)


def start_vm(zone):
    rc, out = gcloud("compute", "instances", "start", VM_NAME, f"--zone={zone}")
    return rc == 0, out


# ─── VM-side probes ────────────────────────────────────────────────────────

def wait_for_ssh(zone, attempts=SSH_ATTEMPTS):
    for i in range(attempts):
        rc, out = ssh(zone, "echo ssh_ready", timeout=30)
        if rc == 0 and "ssh_ready" in out:
            return True
        print(f"{now()} ssh not ready ({i+1}/{attempts}), waiting {SSH_BACKOFF}s",
              flush=True)
        time.sleep(SSH_BACKOFF)
    return False


def wait_for_ready_marker(zone, attempts=READY_MARKER_ATTEMPTS):
    """Poll for /tmp/evo-ready written by vm_startup.sh after base env install.

    Also watches VM status — returns False early if the VM gets preempted
    (TERMINATED) so the outer reconcile loop can restart it without
    wasting the full wait window.
    """
    for i in range(attempts):
        # Bail out fast if the VM has been preempted mid-install.
        status = vm_status(zone)
        if status != "RUNNING":
            print(f"{now()} VM status={status} during ready-marker wait; aborting",
                  flush=True)
            return False

        rc, out = ssh(zone, "test -f /tmp/evo-ready && echo ready || echo not_yet",
                      timeout=30)
        if "ready" in out and "not_yet" not in out:
            print(f"{now()} ✅ startup-script finished base env install",
                  flush=True)
            return True
        print(f"{now()} waiting for /tmp/evo-ready ({i+1}/{attempts})", flush=True)
        time.sleep(READY_MARKER_BACKOFF)
    return False


def repo_uploaded(zone):
    rc, out = ssh(zone,
                  "test -f ~/evo-reward/scripts/run_experiment_jax.py "
                  "&& test -f ~/phase1a_loop.sh && echo ok || echo missing",
                  timeout=30)
    return "ok" in out and "missing" not in out


def tmux_running(zone):
    rc, out = ssh(zone,
                  "tmux has-session -t phase1a 2>/dev/null "
                  "&& echo running || echo stopped",
                  timeout=30)
    return "running" in out and "stopped" not in out


# ─── Upload + launch ───────────────────────────────────────────────────────

def upload_repo(zone, seed, config, runtime):
    tar_path = "/tmp/evo-reward-upload.tar.gz"
    print(f"{now()} tarring local repo -> {tar_path}", flush=True)
    parent = os.path.dirname(REPO_ROOT)
    subprocess.run(
        ["tar",
         "--exclude=evo-reward/.git",
         "--exclude=evo-reward/__pycache__",
         "--exclude=evo-reward/**/__pycache__",
         "--exclude=evo-reward/results",
         "--exclude=evo-reward/dashboard/site/node_modules",
         "--exclude=evo-reward/dashboard/site/dist",
         "--exclude=evo-reward/.venv",
         "--exclude=evo-reward/.pytest_cache",
         "-czf", tar_path, "evo-reward"],
        cwd=parent, check=True,
    )

    print(f"{now()} scp tar -> VM", flush=True)
    rc, out = scp_up(zone, tar_path, "~/evo-reward-new.tar.gz")
    if rc != 0:
        print(f"{now()} scp tar failed: {out.strip()[-300:]}", flush=True)
        return False

    # Write phase1a retry loop script with interpolated args.
    # python -u + stdbuf forces line-buffered output so we see progress
    # lines in phase1a.log in real time instead of waiting for a 4KB
    # block buffer to fill.
    loop_script = f"""#!/bin/bash
source ~/evo-env/bin/activate
cd ~/evo-reward
CKPT_DIR="results/baseline_faithful/seed_{seed}/checkpoints"
while true; do
  if ls $CKPT_DIR/step_*.npz >/dev/null 2>&1; then
    RESUME="--resume"
  else
    RESUME=""
  fi
  stdbuf -oL -eL python -u scripts/run_experiment_jax.py \\
    --config {config} \\
    --runtime {runtime} \\
    --seed {seed} $RESUME 2>&1 | stdbuf -oL tee -a ~/phase1a.log
  rc=$?
  if [ $rc -eq 0 ]; then
    echo "[phase1a] completed at $(date)" | tee -a ~/phase1a.log
    break
  fi
  echo "[phase1a] exit $rc at $(date), retrying in 30s" | tee -a ~/phase1a.log
  sleep 30
done
"""
    loop_path = "/tmp/_evo_phase1a_loop.sh"
    with open(loop_path, "w") as f:
        f.write(loop_script)
    os.chmod(loop_path, 0o755)
    rc, out = scp_up(zone, loop_path, "~/phase1a_loop.sh")
    if rc != 0:
        print(f"{now()} scp loop script failed: {out.strip()[-300:]}", flush=True)
        return False

    print(f"{now()} extracting tar + preserving results/", flush=True)
    merge = (
        "set -e && "
        "rm -rf ~/evo-reward-new && "
        "mkdir ~/evo-reward-new && "
        "tar -xzf ~/evo-reward-new.tar.gz -C ~/evo-reward-new --strip-components=1 && "
        "rm ~/evo-reward-new.tar.gz && "
        "if [ -d ~/evo-reward/results ]; then "
        "  cp -r ~/evo-reward/results ~/evo-reward-new/; "
        "fi && "
        "rm -rf ~/evo-reward && "
        "mv ~/evo-reward-new ~/evo-reward && "
        "chmod +x ~/phase1a_loop.sh && "
        "source ~/evo-env/bin/activate && "
        "cd ~/evo-reward && pip install -q -r requirements.txt && "
        "echo MERGE_DONE"
    )
    rc, out = ssh(zone, merge, timeout=600)
    if "MERGE_DONE" not in out:
        print(f"{now()} merge failed: {out.strip()[-400:]}", flush=True)
        return False
    return True


def start_tmux(zone):
    # Also (re-)start the gcs-sync sidecar here rather than relying on
    # vm_startup.sh. The startup-script runs on boot, which is BEFORE the
    # orchestrator uploads the repo — so when it checks for the repo it
    # finds nothing and skips both phase1a and gcs-sync. We're past that
    # ordering problem now; repo + bucket are in place, so just start it.
    launch = (
        # phase1a training loop
        "tmux kill-session -t phase1a 2>/dev/null; "
        "tmux new-session -d -s phase1a -c ~/evo-reward 'bash ~/phase1a_loop.sh' && "
        # gcs-sync sidecar: pushes results/ to the GCS bucket every 5 min
        f"if ! tmux has-session -t gcs-sync 2>/dev/null; then "
        f"  tmux new-session -d -s gcs-sync -c ~/evo-reward "
        f"    'while true; do "
        f"       gsutil -m rsync -r results gs://{GCS_BUCKET}/results 2>>~/gcs-sync.log; "
        f"       sleep 300; "
        f"     done'; "
        f"fi && "
        "tmux ls"
    )
    rc, out = ssh(zone, launch, timeout=60)
    if rc == 0 and "phase1a" in out:
        sync_state = "✅" if "gcs-sync" in out else "⚠ (no gcs-sync)"
        print(f"{now()} ✅ tmux phase1a session started  gcs-sync: {sync_state}",
              flush=True)
        return True
    print(f"{now()} tmux start failed: {out.strip()[-300:]}", flush=True)
    return False


# ─── Main reconcile loop ──────────────────────────────────────────────────

def reconcile(zone, seed, config, runtime):
    """Drive the VM to the target state. Returns a status string."""
    status = vm_status(zone)
    print(f"{now()} vm_status={status}", flush=True)

    if status == "UNKNOWN":
        return "missing"

    if status == "TERMINATED":
        print(f"{now()} VM terminated — attempting restart in {zone}", flush=True)
        ok, out = start_vm(zone)
        if not ok:
            print(f"{now()} restart failed: {out.strip()[-300:]}", flush=True)
            return "stockout"
        print(f"{now()} restart issued; waiting for RUNNING", flush=True)
        time.sleep(30)
        status = vm_status(zone)

    if status in ("STAGING", "PROVISIONING", "REPAIRING"):
        print(f"{now()} vm transitioning ({status}); wait for next cycle",
              flush=True)
        return "transitioning"

    if status != "RUNNING":
        return "unknown_state"

    # VM is RUNNING — reconcile install state
    if not wait_for_ssh(zone, attempts=3):
        return "ssh_unreachable"

    if not wait_for_ready_marker(zone, attempts=READY_MARKER_ATTEMPTS):
        return "bootstrap_incomplete"

    if not repo_uploaded(zone):
        if not upload_repo(zone, seed, config, runtime):
            return "upload_failed"

    if not tmux_running(zone):
        if not start_tmux(zone):
            return "tmux_failed"

    return "ok"


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default="configs/baseline_faithful.yaml")
    ap.add_argument("--runtime", default="configs/runtime/gcp_l4_spot.yaml")
    ap.add_argument("--on-demand", action="store_true",
                    help="Use on-demand VMs instead of spot. No preemption, ~3x cost.")
    args = ap.parse_args()

    ensure_bucket()
    state = load_state()
    zone = state.get("zone")
    stockout_streak = 0

    try:
        while True:
            try:
                if zone is None:
                    zone = poll_for_capacity(spot=not args.on_demand)
                    state = {"zone": zone, "seed": args.seed, "started_at": now()}
                    save_state(state)
                    stockout_streak = 0
                    mode = "on-demand" if args.on_demand else "spot"
                    print(f"{now()} === provisioning new {mode} VM in {zone} ===",
                          flush=True)

                result = reconcile(zone, args.seed, args.config, args.runtime)
                print(f"{now()} reconcile result: {result}", flush=True)

                if result == "stockout":
                    stockout_streak += 1
                    print(f"{now()} stockout streak: {stockout_streak}/"
                          f"{MAX_CONSECUTIVE_STOCKOUTS}", flush=True)
                    if stockout_streak >= MAX_CONSECUTIVE_STOCKOUTS:
                        print(f"{now()} zone {zone} persistently stocked out — "
                              f"deleting VM and falling back to multi-zone poll",
                              flush=True)
                        gcloud("compute", "instances", "delete", VM_NAME,
                               f"--zone={zone}", "--quiet")
                        zone = None
                        state = {}
                        save_state(state)
                        stockout_streak = 0
                        continue
                else:
                    stockout_streak = 0

                if result == "missing":
                    print(f"{now()} VM gone — polling for new capacity",
                          flush=True)
                    zone = None
                    state = {}
                    save_state(state)
                    continue

                if result == "ok":
                    # Happy path. Print helpful monitoring commands once.
                    if not state.get("_printed_tips"):
                        print(
                            f"{now()} training is live.\n"
                            f"  log:  gcloud compute ssh {VM_NAME} "
                            f"--zone={zone} --tunnel-through-iap "
                            f"--command='tail -f ~/phase1a.log'\n"
                            f"  tmux: gcloud compute ssh {VM_NAME} "
                            f"--zone={zone} --tunnel-through-iap -- "
                            f"-t 'tmux attach -t phase1a'",
                            flush=True,
                        )
                        state["_printed_tips"] = True
                        save_state(state)

                time.sleep(MONITOR_INTERVAL)
            except Exception as e:
                print(f"{now()} reconcile error: {type(e).__name__}: {e}",
                      flush=True)
                print(f"{now()} retrying in {MONITOR_INTERVAL}s", flush=True)
                time.sleep(MONITOR_INTERVAL)
    except KeyboardInterrupt:
        print(f"\n{now()} monitor interrupted by user. VM continues on {zone}.",
              flush=True)
        print(f"Re-run to resume monitoring.", flush=True)


if __name__ == "__main__":
    main()
