#!/bin/bash
# launch_sweep_vm.sh — fire-and-forget on-demand L4 VM launcher for sweep runs.
#
# Usage:
#   scripts/launch_sweep_vm.sh <vm_name> <zone> <config_path> <seed> [runtime]
#
# Example:
#   scripts/launch_sweep_vm.sh evo-sweep-mouth us-east1-b \
#     configs/archive/experiments/sweep_mouth_smol.yaml 0
#
# Each invocation creates one independent VM in the given zone, sets up the
# repo, and starts a phase1a tmux session. No state file, no monitor loop —
# this is the simpler counterpart to spot_orchestrator.py for cases where
# preemption recovery isn't needed (on-demand VMs).
#
# Status check after launch:
#   gcloud compute ssh <vm_name> --zone=<zone> --tunnel-through-iap \
#     --command='tail -30 ~/phase1a.log'
#
# Cleanup:
#   gcloud compute instances delete <vm_name> --zone=<zone> --quiet

set -euo pipefail

if [ $# -lt 4 ]; then
  echo "Usage: $0 <vm_name> <zone> <config_path> <seed> [runtime]" >&2
  exit 1
fi

VM_NAME=$1
ZONE=$2
CONFIG=$3
SEED=$4
RUNTIME=${5:-configs/runtime/gcp_l4.yaml}

REGION=${ZONE%-*}
GCS_BUCKET="${GCS_BUCKET:-evo-reward-ckpts}"
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
STARTUP_SCRIPT="$REPO_ROOT/scripts/vm_startup.sh"
PROJECT="${CLOUDSDK_CORE_PROJECT:-evo-reward}"

if [ ! -f "$REPO_ROOT/$CONFIG" ]; then
  echo "ERROR: config not found: $REPO_ROOT/$CONFIG" >&2
  exit 1
fi

EXPERIMENT=$(grep '^experiment_name:' "$REPO_ROOT/$CONFIG" | head -1 | sed 's/^experiment_name:[[:space:]]*//; s/^"//; s/"$//')
if [ -z "$EXPERIMENT" ]; then
  echo "ERROR: could not extract experiment_name from $CONFIG" >&2
  exit 1
fi

# GCP label values: lowercase [a-z0-9_-], max 63 chars.
LABEL_EXPERIMENT=$(echo "$EXPERIMENT" | tr '[:upper:]' '[:lower:]' | tr '.' '_')

echo "=== launch_sweep_vm.sh ==="
echo "  vm:         $VM_NAME"
echo "  zone:       $ZONE  (region $REGION)"
echo "  config:     $CONFIG"
echo "  experiment: $EXPERIMENT"
echo "  seed:       $SEED"
echo "  runtime:    $RUNTIME"
echo

# ─── 1. Cloud Router + NAT (so --no-address VM has outbound) ────────────────
# us-central1 uses unsuffixed names from the original infra; other regions
# get suffixed names so they don't collide.
if [ "$REGION" = "us-central1" ]; then
  ROUTER_NAME="evo-reward-router"
  NAT_NAME="evo-reward-nat"
else
  REGION_SUFFIX=$(echo "$REGION" | sed 's/^us-//; s/^europe-//')
  ROUTER_NAME="evo-reward-router-${REGION_SUFFIX}"
  NAT_NAME="evo-reward-nat-${REGION_SUFFIX}"
fi

if ! gcloud compute routers describe "$ROUTER_NAME" --region="$REGION" --project="$PROJECT" --format="value(name)" >/dev/null 2>&1; then
  echo ">>> creating Cloud Router + NAT in $REGION"
  gcloud compute routers create "$ROUTER_NAME" \
    --region="$REGION" --network=default --project="$PROJECT"
  gcloud compute routers nats create "$NAT_NAME" \
    --router="$ROUTER_NAME" --region="$REGION" --project="$PROJECT" \
    --nat-all-subnet-ip-ranges --auto-allocate-nat-external-ips
else
  echo ">>> Cloud Router + NAT already present in $REGION"
fi

# ─── 2. Create VM ────────────────────────────────────────────────────────────
echo ">>> creating VM $VM_NAME in $ZONE (on-demand L4)"
gcloud compute instances create "$VM_NAME" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --machine-type=g2-standard-8 \
  --accelerator=type=nvidia-l4,count=1 \
  --image-family=common-cu129-ubuntu-2204-nvidia-580 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB \
  --maintenance-policy=TERMINATE \
  --metadata="install-nvidia-driver=True,gcs-bucket=${GCS_BUCKET}" \
  --metadata-from-file=startup-script="$STARTUP_SCRIPT" \
  --scopes=https://www.googleapis.com/auth/cloud-platform \
  --no-address \
  --labels="experiment=${LABEL_EXPERIMENT},phase=1a,seed=${SEED}"

# ─── 3. Wait for SSH ─────────────────────────────────────────────────────────
echo ">>> waiting for SSH (up to ~10 min)"
for i in $(seq 1 20); do
  if gcloud compute ssh "$VM_NAME" --zone="$ZONE" --project="$PROJECT" \
       --tunnel-through-iap --command='echo ssh_ready' --quiet 2>/dev/null \
       | grep -q ssh_ready; then
    echo "    ssh ready (attempt $i)"
    break
  fi
  echo "    ssh not ready (${i}/20), waiting 30s"
  sleep 30
done

# ─── 4. Wait for /tmp/evo-ready (vm_startup.sh finished base env) ────────────
echo ">>> waiting for vm_startup.sh to finish base env install (up to ~15 min)"
for i in $(seq 1 30); do
  out=$(gcloud compute ssh "$VM_NAME" --zone="$ZONE" --project="$PROJECT" \
          --tunnel-through-iap \
          --command='test -f /tmp/evo-ready && echo ready || echo not_yet' \
          --quiet 2>/dev/null || true)
  if echo "$out" | grep -q '^ready$'; then
    echo "    vm_startup ready (attempt $i)"
    break
  fi
  echo "    not ready (${i}/30), waiting 30s"
  sleep 30
done

# ─── 5. Tar + upload repo ────────────────────────────────────────────────────
TAR_PATH=/tmp/evo-reward-upload-${VM_NAME}.tar.gz
echo ">>> tarring repo -> $TAR_PATH"
tar \
  --exclude='evo-reward/.git' \
  --exclude='evo-reward/__pycache__' \
  --exclude='evo-reward/results' \
  --exclude='evo-reward/dashboard/site/node_modules' \
  --exclude='evo-reward/dashboard/site/dist' \
  --exclude='evo-reward/.venv' \
  --exclude='evo-reward/.pytest_cache' \
  -czf "$TAR_PATH" -C "$(dirname "$REPO_ROOT")" evo-reward

echo ">>> scp tar to VM"
gcloud compute scp "$TAR_PATH" "$VM_NAME:~/evo-reward-new.tar.gz" \
  --zone="$ZONE" --project="$PROJECT" --tunnel-through-iap

# ─── 6. Write per-experiment loop script (fixes hardcoded ckpt path bug) ────
LOOP_SCRIPT=/tmp/_sweep_phase1a_loop_${VM_NAME}.sh
cat > "$LOOP_SCRIPT" <<EOF
#!/bin/bash
source ~/evo-env/bin/activate
cd ~/evo-reward
CKPT_DIR="results/${EXPERIMENT}/seed_${SEED}/checkpoints"
while true; do
  if ls "\$CKPT_DIR"/step_*.npz >/dev/null 2>&1; then
    RESUME="--resume"
  else
    RESUME=""
  fi
  stdbuf -oL -eL python -u scripts/run_experiment_jax.py \\
    --config ${CONFIG} \\
    --runtime ${RUNTIME} \\
    --seed ${SEED} \$RESUME 2>&1 | stdbuf -oL tee -a ~/phase1a.log
  rc=\$?
  if [ \$rc -eq 0 ]; then
    echo "[phase1a] completed at \$(date)" | tee -a ~/phase1a.log
    break
  fi
  echo "[phase1a] exit \$rc at \$(date), retrying in 30s" | tee -a ~/phase1a.log
  sleep 30
done
EOF
chmod +x "$LOOP_SCRIPT"
gcloud compute scp "$LOOP_SCRIPT" "$VM_NAME:~/phase1a_loop.sh" \
  --zone="$ZONE" --project="$PROJECT" --tunnel-through-iap

# ─── 7. Extract + install requirements + start tmux ──────────────────────────
echo ">>> extracting tar, installing requirements, starting tmux sessions"
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --project="$PROJECT" --tunnel-through-iap --command="
set -e
rm -rf ~/evo-reward-new
mkdir ~/evo-reward-new
tar -xzf ~/evo-reward-new.tar.gz -C ~/evo-reward-new --strip-components=1
rm ~/evo-reward-new.tar.gz
if [ -d ~/evo-reward/results ]; then
  cp -r ~/evo-reward/results ~/evo-reward-new/
fi
rm -rf ~/evo-reward
mv ~/evo-reward-new ~/evo-reward
chmod +x ~/phase1a_loop.sh
source ~/evo-env/bin/activate
cd ~/evo-reward
pip install -q -r requirements.txt

tmux kill-session -t phase1a 2>/dev/null || true
tmux new-session -d -s phase1a -c ~/evo-reward 'bash ~/phase1a_loop.sh'

tmux kill-session -t gcs-sync 2>/dev/null || true
tmux new-session -d -s gcs-sync -c ~/evo-reward \
  'while true; do gsutil -m rsync -r results gs://${GCS_BUCKET}/results 2>>~/gcs-sync.log; sleep 300; done'

echo
echo 'tmux sessions:'
tmux ls
"

cat <<EOF

============================================================
VM $VM_NAME launched in $ZONE
  experiment: $EXPERIMENT
  seed:       $SEED
  config:     $CONFIG

  Tail log:
    gcloud compute ssh $VM_NAME --zone=$ZONE --project=$PROJECT \\
      --tunnel-through-iap --command='tail -30 ~/phase1a.log'

  Attach tmux:
    gcloud compute ssh $VM_NAME --zone=$ZONE --project=$PROJECT \\
      --tunnel-through-iap -- -t 'tmux attach -t phase1a'

  Delete VM (when done):
    gcloud compute instances delete $VM_NAME --zone=$ZONE \\
      --project=$PROJECT --quiet
============================================================
EOF
