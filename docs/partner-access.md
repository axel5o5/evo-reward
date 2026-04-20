# Collaborator access — `evo-reward` GCP project

Reference for a collaborator with the following IAM on the `evo-reward`
project:

- `roles/viewer` — see any resource
- `roles/iap.tunnelResourceAccessor` — SSH via IAP
- `roles/compute.osLogin` — SSH key provisioning
- `roles/storage.objectViewer` — read checkpoint bucket contents

These cover everything a non-admin collaborator needs: inspect the VM,
SSH in to poke around, download checkpoints. You cannot start / stop /
delete VMs or modify anything — that's intentional. If you need more,
ask Axel.

---

## TL;DR — just watch what's running

Open the dashboard. No gcloud needed.

```
https://evo-reward.axelinter.net/gcp
```

Ask Axel for the PIN if you want to see the Stop/Start/Restart/Delete
buttons (viewing is anonymous).

The dashboard shows live VM status, checkpoint freshness, cost
estimate, and full training progress (step / rate / ETA / population /
8 reward weights + evolution-detected flag).

---

## One-time local setup

Install `gcloud`:

```bash
# macOS
brew install --cask google-cloud-sdk

# Or: https://cloud.google.com/sdk/docs/install
```

Isolate this project in its own gcloud configuration so it doesn't
clash with any other GCP projects on your machine:

```bash
gcloud config configurations create evo-reward
gcloud auth login <your-email>
gcloud config set account <your-email>
gcloud config set project evo-reward

# Switch back to any other config later:
#   gcloud config configurations activate <name>
```

For the Python libraries used by tooling (optional), set up Application
Default Credentials:

```bash
gcloud auth application-default login
```

Verify:

```bash
gcloud config list
gcloud compute instances list
# Expected: one row — evo-reward-gpu in some us-west/us-central zone
```

---

## Common commands

### See the VM

```bash
# List with status + zone
gcloud compute instances list

# Full details (machine type, labels, IP, disks…)
gcloud compute instances describe evo-reward-gpu \
  --zone=$(gcloud compute instances list --filter="name=evo-reward-gpu" --format="value(zone)")
```

The `labels` field tells you what run is on the VM (`experiment`,
`phase`, `seed`). It matches the configs under `configs/` in the repo.

### SSH via IAP (no public IP needed)

```bash
# Replace ZONE with whatever `gcloud compute instances list` shows.
ZONE=us-west1-a

gcloud compute ssh evo-reward-gpu --zone=$ZONE --tunnel-through-iap
```

Once connected, the training runs in tmux. Don't kill it. Useful
commands from the VM:

```bash
# Attach (read-only recommended — Ctrl-b d to detach; NEVER Ctrl-c or you kill training)
tmux attach -t phase1a

# Safer: tail the log non-interactively
tail -f ~/phase1a.log

# GPU utilization
nvidia-smi
```

### Download checkpoints or progress.json

The bucket has all checkpoint `.npz` files + the live `progress.json`.

```bash
# Recent progress (same data the dashboard shows)
gcloud storage cat gs://evo-reward-ckpts/results/baseline_faithful/seed_0/progress.json | jq

# List checkpoints
gcloud storage ls gs://evo-reward-ckpts/results/baseline_faithful/seed_0/checkpoints/

# Download the latest checkpoint locally
gcloud storage cp gs://evo-reward-ckpts/results/baseline_faithful/seed_0/checkpoints/step_XXXXX.npz ./
```

### Tail the training log remotely (no SSH session)

One-liner, exits when the VM is idle:

```bash
gcloud compute ssh evo-reward-gpu --zone=$ZONE --tunnel-through-iap \
  --command='tail -n 40 ~/phase1a.log'
```

---

## The local status tool

The repo has a compact status script that polls the VM and prints a
single-screen summary. Useful if you want terminal-native monitoring
instead of the browser dashboard.

```bash
# From a clone of the repo:
cd evo-reward
source scripts/gcloud-env.sh      # (you may need to edit it to use your account)
python3 scripts/status.py

# Live, refreshes every 30s:
python3 scripts/status.py --follow 30
```

See [docs/monitoring.md](monitoring.md) for what each line means.

---

## What you cannot do (and why)

With the current IAM:

| Action | Why it's blocked |
|---|---|
| Stop / start / delete the VM | No `compute.instanceAdmin` role |
| Create new VMs or disks | No admin role |
| Modify IAM or billing | Not an admin |
| Write to the GCS bucket | `storage.objectViewer` is read-only |

If something looks broken and you want to intervene (e.g. restart the
training tmux session), ask Axel — or use the dashboard's control
panel if you have the PIN.

---

## Troubleshooting

**`Permission denied` on SSH** — your first IAP SSH may fail with
"failed to connect to backend." Wait 30 seconds and retry; the IAP
daemon takes a moment to register a new identity.

**`Required 'compute.instances.start' permission`** on Start/Stop —
expected. Viewer role can't do mutations. Use the dashboard if you
have the PIN; otherwise ask Axel.

**`ZONE_RESOURCE_POOL_EXHAUSTED`** — not relevant to you unless you
try to create resources, which you can't.

**Dashboard won't load** — the `/gcp` page needs no auth to view
monitor data. If it fails to fetch, the monitor workflow probably
hasn't run recently. Check GitHub Actions → GCP Monitor.
