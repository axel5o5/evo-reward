# Phase 1a on GCP Spot L4 (cheap, interruptible)

This is a cost-saving addendum to [gcp-setup.md](gcp-setup.md). It runs
Phase 1a on a **spot/preemptible L4 VM** — same hardware at ~70% off,
at the price of occasional preemptions that the runner's `--resume`
flow absorbs.

Before reading this, provision the shared infrastructure (project,
APIs, quotas, Cloud NAT, IAP firewall) using steps 1–6 of
[gcp-setup.md](gcp-setup.md). Only the VM-create and training-loop
steps change.

---

## ⚠ When spot works — and when it doesn't

**Lessons from the April 2026 Phase 1a seed 0 run.** Spot capacity for
L4s was persistently contested across all 9 zones we rotate through.
Observed failure pattern:

- ~6 consecutive spot VMs were preempted within 10–15 min of creation
- None survived long enough to save a checkpoint to GCS
- All training progress from each attempt was lost
- Eventually fell back to `--on-demand` for guaranteed forward progress
  (~3× the cost)

**Decision rule for any future run.** Start with spot if budget matters,
but watch the first VM closely. If it's preempted within 15 min of
creation, that's a strong signal capacity is contested — switch to
on-demand rather than bleeding time on retries.

```bash
# One-liner to switch: kill the spot orchestrator, relaunch on-demand.
# GCS-backed checkpoints carry over automatically.
pkill -f spot_orchestrator
caffeinate -d -i python3 scripts/spot_orchestrator.py --seed 0 --on-demand
```

The cost numbers below assume spot actually survives multi-hour runs.
In contested-capacity periods, the true cost can exceed on-demand
because you pay for repeated provisioning without forward progress.
See [docs/monitoring.md](monitoring.md#telling-if-spot-isnt-going-to-work)
for what "contested capacity" looks like from the orchestrator logs.

---

## Cost comparison (single seed of Phase 1a, ~12h nominal)

| Path | $/hr | Typical wall clock | Compute | NAT | **Total** |
| --- | --- | --- | --- | --- | --- |
| On-demand L4 | $0.70 | 12h | $8.40 | $0.50 | **$8.90** |
| Spot L4 (+20% preemption overhead) | $0.21 | 14.4h | $3.02 | $0.60 | **$3.62** |
| Spot L4 (worst case, 2× slower) | $0.21 | 24h | $5.04 | $1.00 | **$6.04** |

Savings: ~$3–5 per seed, ~$15–25 for a 5-seed run.

---

## What preemption actually looks like

1. GCP sends your VM `SIGTERM` with a 30-second grace period.
2. VM is **STOPPED** (not deleted — the disk survives).
3. You stop being billed for compute; disk billing continues (~$0.02/hr).
4. The VM can be restarted with `gcloud compute instances start ...`
   once spot capacity returns (usually minutes, sometimes hours).
5. On restart, the runner's `--resume` picks up from the last
   checkpoint — at most 25K steps (~1.75 min) lost.

---

## 1. Create the spot VM

Same as step 7 of gcp-setup.md, with one new flag:

```
gcloud compute instances create evo-reward-gpu-spot \
  --project=evo-reward \
  --zone=us-central1-b \
  --machine-type=g2-standard-8 \
  --accelerator=type=nvidia-l4,count=1 \
  --image-family=common-cu129-ubuntu-2204-nvidia-580 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB \
  --maintenance-policy=TERMINATE \
  --metadata="install-nvidia-driver=True" \
  --no-address \
  --provisioning-model=SPOT \
  --instance-termination-action=STOP
```

Key differences vs on-demand:

- `--provisioning-model=SPOT` — opts into spot pricing.
- `--instance-termination-action=STOP` — preservation of disk on
  preemption (vs `DELETE`, which throws it away).
- Name `evo-reward-gpu-spot` to keep it separate from any on-demand VM.

---

## 2. SSH and one-time setup

```
gcloud compute ssh evo-reward-gpu-spot \
  --zone=us-central1-b --project=evo-reward \
  --tunnel-through-iap
```

Then run the same install steps as gcp-setup.md step 8:

```
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv git tmux
python3 -m venv ~/evo-env
source ~/evo-env/bin/activate
pip install -U pip "jax[cuda12]"
```

Upload the repo (step 9 of gcp-setup.md, `gcloud compute scp` via
IAP), then:

```
cd ~/evo-reward
source ~/evo-env/bin/activate
pip install -r requirements.txt
```

---

## 3. Run with the retry loop

Spot VMs can be preempted while your Python process is running. The
runner's `--resume` handles picking up from the last checkpoint, but
something has to **re-invoke the runner** after the VM boots back up.
Easiest way: a shell retry loop inside tmux.

```
tmux new -s phase1a
source ~/evo-env/bin/activate
cd ~/evo-reward

while true; do
  python scripts/run_experiment_jax.py \
    --config configs/baseline_faithful.yaml \
    --runtime configs/runtime/gcp_l4_spot.yaml \
    --seed 0 \
    --resume
  rc=$?
  if [ $rc -eq 0 ]; then
    echo "[$(date)] Run completed (exit 0)."
    break
  fi
  echo "[$(date)] Run exited with code $rc — sleeping 30s then retrying"
  sleep 30
done
```

Detach with `Ctrl-b d`. Because this is inside tmux, the loop
survives SSH disconnects. It does **not** survive VM preemption —
tmux dies with the VM.

The first invocation (fresh start) won't find a checkpoint, so
`--resume` would error. Run once without `--resume`, then modify the
loop. Or seed with `--max-steps 1` first to create an initial
checkpoint, then start the loop.

---

## 4. Handling preemption

When GCP preempts the VM:

1. `gcloud compute instances describe evo-reward-gpu-spot ...
   --format='value(status)'` returns `TERMINATED`.
2. Start it back up: `gcloud compute instances start
   evo-reward-gpu-spot --zone=us-central1-b --project=evo-reward`.
3. SSH back in, `tmux attach -t phase1a` — your old session is gone
   (VM rebooted). Rerun the retry loop; `--resume` picks up from the
   last checkpoint.

To automate step 1–2 from your laptop:

```
# Simple poll-and-restart loop
while true; do
  status=$(gcloud compute instances describe evo-reward-gpu-spot \
    --zone=us-central1-b --project=evo-reward \
    --format='value(status)' 2>/dev/null)
  if [ "$status" = "TERMINATED" ]; then
    echo "[$(date)] VM preempted — restarting"
    gcloud compute instances start evo-reward-gpu-spot \
      --zone=us-central1-b --project=evo-reward
    sleep 90  # wait for boot + driver init
    # Kick off the retry loop again
    gcloud compute ssh evo-reward-gpu-spot \
      --zone=us-central1-b --project=evo-reward --tunnel-through-iap \
      --command "tmux new-session -d -s phase1a \
        'source ~/evo-env/bin/activate && cd ~/evo-reward && \
         while true; do \
           python scripts/run_experiment_jax.py \
             --config configs/baseline_faithful.yaml \
             --runtime configs/runtime/gcp_l4_spot.yaml \
             --seed 0 --resume; \
           [ \$? -eq 0 ] && break; sleep 30; \
         done'"
  fi
  sleep 60
done
```

This isn't beautiful but it's practical. A cleaner solution is a
systemd unit on the VM that auto-starts the retry loop on boot —
worth doing if you find yourself running many seeds.

---

## 5. Watch progress / download results

Same as on-demand (gcp-setup.md steps 10–11). The checkpoint path is
`~/evo-reward/results/baseline_faithful/seed_0/checkpoints/` — pull
it to your laptop periodically via `gcloud compute scp --recurse`
in case the whole VM vanishes.

---

## When NOT to use spot

- **Deadline-sensitive runs.** If you need seed 0 to finish by
  Thursday, spot's wall-clock unpredictability is not your friend.
- **Low-capacity periods.** If spot L4 availability is zero in
  your zone (check the GCP console's "Create VM" flow — it'll refuse
  with a clear error), you're stuck waiting anyway. Fall back to
  on-demand.
- **First validation.** For the *very first* seed of a new code
  branch, paying the on-demand premium gets you a clean answer
  fast. Use spot once you trust the run.
