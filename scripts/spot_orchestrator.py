"""
spot_orchestrator.py
--------------------
Orchestrate a spot-L4 Phase 1a run that survives preemptions.

  1. Poll zones for spot L4 capacity (every 10 min) until one accepts.
  2. Provision the VM: apt packages, Python venv, jax[cuda12], repo upload.
  3. Start training in tmux with a retry loop (handles Python-level crashes).
  4. Monitor VM status forever; on STOP/TERMINATED (preemption), retry
     `instances start` in the same zone and re-launch the tmux loop.

State is written to .claude-orch-state.json so re-invocations pick up
an existing run rather than creating a duplicate VM.

Usage:
  source scripts/gcloud-env.sh
  python3 scripts/spot_orchestrator.py --seed 0

Ctrl-C to stop monitoring from your laptop. Training on the VM keeps
running either way (it's in tmux). Re-run the script later to resume
monitoring.
"""

import argparse
import json
import os
import subprocess
import sys
import time


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_FILE = os.path.join(REPO_ROOT, ".claude-orch-state.json")

ZONES_TO_TRY = [
    "us-central1-a", "us-east4-b", "us-east4-c",
    "us-west1-a", "us-west1-b",
    "us-central1-b", "us-central1-c",
    "us-east1-b", "us-east1-c",
]

VM_NAME = "evo-reward-gpu"
ROUTER_PREFIX = "evo-reward-router"
NAT_PREFIX = "evo-reward-nat"
POLL_INTERVAL = 600         # 10 min between capacity polls
MONITOR_INTERVAL = 600      # 10 min between VM status checks
BOOT_WAIT = 90              # sec after `instances start` before SSH
SSH_RETRY_ATTEMPTS = 6      # ~7 min total (30+60+90+120+150+180)
SSH_RETRY_BACKOFF = [30, 60, 90, 120, 150, 180]


def region_of(zone: str) -> str:
    """us-central1-a -> us-central1"""
    return zone.rsplit("-", 1)[0]


def now() -> str:
    return time.strftime("[%Y-%m-%d %H:%M:%S]")


def gcloud(*args, timeout=300):
    """Run a gcloud command, return (rc, combined_output)."""
    cmd = ["gcloud"] + list(args)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired:
        return -1, "TIMEOUT"


def ssh_exec(zone, remote_cmd, timeout=600):
    return gcloud(
        "compute", "ssh", VM_NAME,
        f"--zone={zone}", "--tunnel-through-iap",
        f"--command={remote_cmd}",
        timeout=timeout,
    )


def scp_up(zone, local_path, remote_path):
    return gcloud(
        "compute", "scp", local_path, f"{VM_NAME}:{remote_path}",
        f"--zone={zone}", "--tunnel-through-iap",
        timeout=600,
    )


def ensure_nat(zone):
    """Create Cloud Router + NAT in the region of `zone` if missing.

    NAT is required for VMs with --no-address to reach the internet
    (pip, git, etc). Idempotent: no-ops if the router already exists.
    """
    region = region_of(zone)
    router = f"{ROUTER_PREFIX}-{region}" if region != "us-central1" else ROUTER_PREFIX
    nat = f"{NAT_PREFIX}-{region}" if region != "us-central1" else NAT_PREFIX

    # Check router exists
    rc, _ = gcloud(
        "compute", "routers", "describe", router,
        f"--region={region}", "--format=value(name)",
    )
    if rc == 0:
        return  # router (and presumably NAT) already there

    print(f"{now()} creating Cloud Router + NAT in {region}", flush=True)
    rc, out = gcloud(
        "compute", "routers", "create", router,
        f"--region={region}", "--network=default",
    )
    if rc != 0 and "already exists" not in out:
        print(f"{now()} router create failed: {out.strip()[-300:]}", flush=True)
        return

    rc, out = gcloud(
        "compute", "routers", "nats", "create", nat,
        f"--router={router}", f"--region={region}",
        "--nat-all-subnet-ip-ranges",
        "--auto-allocate-nat-external-ips",
    )
    if rc != 0 and "already exists" not in out:
        print(f"{now()} NAT create failed: {out.strip()[-300:]}", flush=True)


def wait_for_ssh(zone, attempts=SSH_RETRY_ATTEMPTS):
    """Poll SSH until it connects, with exponential backoff. Returns True on success."""
    for i, wait in enumerate(SSH_RETRY_BACKOFF[:attempts]):
        print(f"{now()} waiting {wait}s for SSH readiness (attempt {i+1}/{attempts})", flush=True)
        time.sleep(wait)
        rc, out = ssh_exec(zone, "echo ssh_ready", timeout=60)
        if rc == 0 and "ssh_ready" in out:
            print(f"{now()} ✅ SSH ready", flush=True)
            return True
    print(f"{now()} SSH never became ready after {attempts} attempts", flush=True)
    return False


def try_create_spot(zone):
    print(f"{now()} trying {zone}...", flush=True)
    rc, out = gcloud(
        "compute", "instances", "create", VM_NAME,
        f"--zone={zone}",
        "--machine-type=g2-standard-8",
        "--accelerator=type=nvidia-l4,count=1",
        "--image-family=common-cu129-ubuntu-2204-nvidia-580",
        "--image-project=deeplearning-platform-release",
        "--boot-disk-size=200GB",
        "--maintenance-policy=TERMINATE",
        "--metadata=install-nvidia-driver=True",
        "--no-address",
        "--provisioning-model=SPOT",
        "--instance-termination-action=STOP",
    )
    ok = rc == 0 and "RUNNING" in out
    if ok:
        print(f"{now()} ✅ capacity in {zone}", flush=True)
        ensure_nat(zone)
        wait_for_ssh(zone)
    return ok


def poll_for_capacity():
    while True:
        for zone in ZONES_TO_TRY:
            if try_create_spot(zone):
                return zone
        print(f"{now()} all zones stockout, sleeping {POLL_INTERVAL}s", flush=True)
        time.sleep(POLL_INTERVAL)


def provision(zone):
    print(f"{now()} provisioning: apt + venv + jax[cuda12]", flush=True)
    setup = (
        "set -e && "
        "sudo apt-get update -qq && "
        "sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "
        "python3-pip python3-venv git tmux >/dev/null && "
        "[ -d ~/evo-env ] || python3 -m venv ~/evo-env && "
        "source ~/evo-env/bin/activate && "
        "pip install -q -U pip 'jax[cuda12]' && "
        "python -c 'import jax; print(\"devices=\", jax.devices())'"
    )
    rc, out = ssh_exec(zone, setup, timeout=900)
    print(out.strip()[-400:], flush=True)
    if rc != 0:
        print(f"{now()} setup failed (rc={rc})", file=sys.stderr, flush=True)
        sys.exit(1)

    # Tar current repo (excluding heavy / ephemeral dirs)
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
        cwd=parent,
        check=True,
    )
    print(f"{now()} scp upload...", flush=True)
    rc, out = scp_up(zone, tar_path, "~/evo-reward-new.tar.gz")
    if rc != 0:
        print(f"{now()} scp failed (rc={rc}): {out}", file=sys.stderr, flush=True)
        sys.exit(1)

    print(f"{now()} extracting + preserving existing results/", flush=True)
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
        "cd ~/evo-reward && "
        "source ~/evo-env/bin/activate && "
        "pip install -q -r requirements.txt"
    )
    rc, out = ssh_exec(zone, merge, timeout=900)
    if rc != 0:
        print(f"{now()} merge/install failed: {out}", file=sys.stderr, flush=True)
        sys.exit(1)


def start_training(zone, seed, config, runtime):
    """Write remote loop script, scp it, kick off tmux session."""
    print(f"{now()} (re)starting tmux training loop", flush=True)
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
  python scripts/run_experiment_jax.py \\
    --config {config} \\
    --runtime {runtime} \\
    --seed {seed} $RESUME 2>&1 | tee -a ~/phase1a.log
  rc=$?
  if [ $rc -eq 0 ]; then
    echo "[phase1a] completed at $(date)" | tee -a ~/phase1a.log
    break
  fi
  echo "[phase1a] exit $rc at $(date), retrying in 30s" | tee -a ~/phase1a.log
  sleep 30
done
"""
    local_path = "/tmp/_evo_phase1a_loop.sh"
    with open(local_path, "w") as f:
        f.write(loop_script)
    os.chmod(local_path, 0o755)
    rc, out = scp_up(zone, local_path, "~/phase1a_loop.sh")
    if rc != 0:
        print(f"{now()} scp loop script failed: {out}", file=sys.stderr, flush=True)
        sys.exit(1)

    launch = (
        "chmod +x ~/phase1a_loop.sh && "
        "tmux kill-session -t phase1a 2>/dev/null; "
        "tmux new-session -d -s phase1a '~/phase1a_loop.sh' && "
        "tmux ls"
    )
    rc, out = ssh_exec(zone, launch, timeout=60)
    print(out.strip(), flush=True)
    if rc != 0:
        print(f"{now()} tmux launch failed", file=sys.stderr, flush=True)
        sys.exit(1)


def vm_status(zone):
    rc, out = gcloud(
        "compute", "instances", "describe", VM_NAME,
        f"--zone={zone}", "--format=value(status)",
    )
    return out.strip() if rc == 0 else "UNKNOWN"


def start_vm(zone):
    rc, out = gcloud("compute", "instances", "start", VM_NAME, f"--zone={zone}")
    return rc == 0, out


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
    args = ap.parse_args()

    state = load_state()
    zone = None

    if state.get("zone"):
        zone = state["zone"]
        status = vm_status(zone)
        if status in ("RUNNING", "TERMINATED", "STOPPING", "STAGING"):
            print(f"{now()} resuming monitoring for {VM_NAME} in {zone} (status={status})", flush=True)
        else:
            print(f"{now()} VM missing in {zone}, will create fresh", flush=True)
            zone = None
            state = {}

    if zone is None:
        zone = poll_for_capacity()
        state = {"zone": zone, "seed": args.seed, "started_at": now()}
        save_state(state)
        provision(zone)
        start_training(zone, args.seed, args.config, args.runtime)

    print(f"{now()} === Monitoring ===", flush=True)
    print(f"VM: {VM_NAME} in {zone}", flush=True)
    print(f"  log:  gcloud compute ssh {VM_NAME} --zone={zone} --tunnel-through-iap --command='tail -f ~/phase1a.log'", flush=True)
    print(f"  tmux: gcloud compute ssh {VM_NAME} --zone={zone} --tunnel-through-iap -- -t 'tmux attach -t phase1a'", flush=True)
    print("", flush=True)

    try:
        while True:
            status = vm_status(zone)
            print(f"{now()} status={status}", flush=True)
            if status == "TERMINATED":
                print(f"{now()} preempted — attempting restart in {zone}", flush=True)
                ok, out = start_vm(zone)
                if ok:
                    print(f"{now()} restarted; waiting {BOOT_WAIT}s for boot", flush=True)
                    time.sleep(BOOT_WAIT)
                    start_training(zone, args.seed, args.config, args.runtime)
                else:
                    # Likely zone stockout. Keep retrying same zone on schedule.
                    print(f"{now()} restart failed: {out.strip()[-300:]}", flush=True)
                    print(f"{now()} will retry in {MONITOR_INTERVAL}s", flush=True)
            time.sleep(MONITOR_INTERVAL)
    except KeyboardInterrupt:
        print(f"\n{now()} monitor interrupted by user. VM training continues on {VM_NAME}/{zone}.", flush=True)
        print(f"Re-run `python3 scripts/spot_orchestrator.py` to resume monitoring.", flush=True)


if __name__ == "__main__":
    main()
