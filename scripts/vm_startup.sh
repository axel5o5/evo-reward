#!/bin/bash
# VM startup-script — runs as root on every VM boot (including after
# spot preemption + restart).
#
# Installs apt packages + Python venv + jax[cuda12]. If the repo has
# been uploaded and a phase1a_loop.sh is present, also starts the
# training tmux session. Idempotent: every step short-circuits if
# already done, so re-running on VM restart is cheap.
#
# Logs to /var/log/evo-startup.log. Writes /tmp/evo-ready when the
# base env is installed so the orchestrator can stop waiting.
#
# Installed as instance metadata via:
#   --metadata-from-file=startup-script=scripts/vm_startup.sh

set -e
exec > /var/log/evo-startup.log 2>&1
echo "=== vm_startup.sh started at $(date) ==="

# The SSH user GCP auto-provisions matches the first home dir. Discover it.
USER_NAME=$(ls /home 2>/dev/null | head -1)
if [ -z "$USER_NAME" ]; then
  echo "no user yet in /home — waiting..."
  for i in {1..60}; do
    USER_NAME=$(ls /home 2>/dev/null | head -1)
    [ -n "$USER_NAME" ] && break
    sleep 2
  done
fi
HOME_DIR="/home/$USER_NAME"
echo "user=$USER_NAME home=$HOME_DIR"

# Wait for nvidia driver (may still be installing on first boot)
for i in {1..60}; do
  if nvidia-smi >/dev/null 2>&1; then
    echo "nvidia driver ready (attempt $i)"
    break
  fi
  echo "waiting for nvidia driver (attempt $i/60)..."
  sleep 5
done

# ── Step 1: apt packages (idempotent; apt-get short-circuits if installed) ──
echo ">>> installing apt packages"
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  python3-pip python3-venv git tmux rsync

# ── Step 2: Python venv + JAX (run as the user) ──
echo ">>> setting up venv + jax[cuda12]"
sudo -u "$USER_NAME" -H bash -lc "
  set -e
  cd $HOME_DIR
  if [ ! -d evo-env ]; then
    python3 -m venv evo-env
  fi
  source evo-env/bin/activate
  pip install -q -U pip
  # Idempotent — pip no-ops if satisfied
  pip install -q 'jax[cuda12]'
"

# Env ready marker — orchestrator polls for this
touch /tmp/evo-ready
chmod 0644 /tmp/evo-ready

# ── Step 3: if repo + loop script are uploaded, install reqs and start training ──
if [ -d "$HOME_DIR/evo-reward" ] && [ -f "$HOME_DIR/phase1a_loop.sh" ]; then
  echo ">>> repo + loop script present; installing requirements"
  sudo -u "$USER_NAME" -H bash -lc "
    set -e
    source $HOME_DIR/evo-env/bin/activate
    cd $HOME_DIR/evo-reward
    pip install -q -r requirements.txt
  "

  # Start tmux if not already running. After preemption restart, the old
  # tmux session is gone but the repo + checkpoints persist on disk, so
  # this re-launches the retry-loop which will pick up from last ckpt.
  if ! sudo -u "$USER_NAME" -H tmux has-session -t phase1a 2>/dev/null; then
    echo ">>> starting phase1a tmux session"
    sudo -u "$USER_NAME" -H bash -lc "
      chmod +x $HOME_DIR/phase1a_loop.sh
      tmux new-session -d -s phase1a -c $HOME_DIR/evo-reward \
        'bash $HOME_DIR/phase1a_loop.sh'
    "
  else
    echo ">>> phase1a tmux session already running"
  fi
else
  echo ">>> repo or loop script not yet present (orchestrator will upload)"
fi

echo "=== vm_startup.sh finished at $(date) ==="
