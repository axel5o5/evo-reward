# Monitoring the GCP Phase 1a Run

Tooling for running + watching Phase 1a on a GCP spot L4 VM. The goal
is that a user never has to remember a raw `gcloud` command for the
common operations (start, check, resume, switch modes, tail log).

The full setup runbook lives in
[`docs/gcp-spot.md`](gcp-spot.md) — this file covers the
day-to-day monitoring layer once the run is going.

---

## 🛑 STOP vs DELETE — read this first

There are two different lifecycle operations that sound similar but are
**very different** in what they destroy.

### You want to PAUSE the VM to save $$$ (keep everything)

Use `stop`. Disk is preserved, including all local checkpoints that
haven't yet been synced to GCS. No compute charges while stopped
(disk storage still costs ~$10/month for 200 GB, negligible).

```bash
source scripts/gcloud-env.sh
gcloud compute instances stop evo-reward-gpu --zone=us-west1-a
```

To bring it back online later, `start` instead of `create`:

```bash
gcloud compute instances start evo-reward-gpu --zone=us-west1-a
# then SSH in; the vm_startup.sh will re-mount, re-sync, and if
# phase1a_loop.sh is present it auto-resumes training.
```

### You are ABSOLUTELY DONE (destroy the VM)

Use `delete`. This is **irreversible for the VM's local disk**.
Anything that hasn't been synced to GCS at the moment of delete is
gone forever.

```bash
# Only run this if you've already pulled everything you want out of GCS
gcloud compute instances delete evo-reward-gpu --zone=us-west1-a --quiet
```

**What gets lost on delete:**
- Local checkpoints under `~/evo-reward/results/` that haven't been
  rsync'd to `gs://evo-reward-ckpts/results/` yet (the gcs-sync tmux
  session only pushes every 5 min — so up to 5 min of training state
  can be lost if you delete mid-sync).
- SSH keys the VM was authenticated with (you'll re-generate on next
  VM).
- tmux session history, `phase1a.log`, any ad-hoc trace files under
  `~/` that aren't inside `results/`.

**What survives a delete:**
- Everything already in `gs://evo-reward-ckpts/` (the checkpoint
  bucket) — replays, progress.json, whatever the sync sidecar pushed.
- Everything in `gs://evo-reward-replays-public/` — all uploaded
  replays (these upload at flush time, every 20K sim steps).
- The GCP project configuration, IAP firewall, NAT router.

### Rule of thumb

If you're not 100% sure you want to lose the VM's disk, **always use
`stop`**. You can always `delete` later from a stopped VM; you can't
un-delete.

---

## Tools at a glance

| Tool | Purpose |
| --- | --- |
| [`scripts/gcloud-env.sh`](../scripts/gcloud-env.sh) | Scope gcloud to the evo-reward project for **this shell only** (doesn't touch global config). `source` before running any of the below. |
| [`scripts/spot_orchestrator.py`](../scripts/spot_orchestrator.py) | Launches and maintains the VM. Polls for spot capacity, handles preemptions, delete-and-recreates on persistent stockout. `--on-demand` flag switches to guaranteed-capacity at ~3× cost. |
| [`scripts/status.py`](../scripts/status.py) | One-shot or live status dashboard. Shows VM state, training progress, reward-weight dynamics, GPU utilization. |
| [`scripts/vm_startup.sh`](../scripts/vm_startup.sh) | Startup script baked into VM metadata. Runs on every boot, installs deps, starts training tmux. Not invoked directly. |

---

## Quick reference

```bash
# At the top of any terminal you use for this project:
source scripts/gcloud-env.sh

# Kick off the run (leave terminal open; caffeinate keeps monitor alive)
caffeinate -d -i python3 scripts/spot_orchestrator.py --seed 0

# Check status (one-shot)
python3 scripts/status.py

# Live dashboard (refreshes every 30s)
python3 scripts/status.py --follow 30

# Tail training log directly (same info, less curated)
gcloud compute ssh evo-reward-gpu --zone=$ZONE --tunnel-through-iap \
  --command='tail -f ~/phase1a.log'

# Attach to the running tmux session
gcloud compute ssh evo-reward-gpu --zone=$ZONE --tunnel-through-iap -- \
  -t 'tmux attach -t phase1a'
```

The `$ZONE` for any command is whatever `status.py` shows — also in
`.claude-orch-state.json`.

---

## What `status.py` shows

Running `python3 scripts/status.py` on a healthy run produces something
like:

```
════════════════════════════════════════════════════════════════════════════════
 Phase 1a — evo-reward-gpu @ 2026-04-20 14:13:00
════════════════════════════════════════════════════════════════════════════════

VM    RUNNING  us-central1-b  g2-standard-8  (spot)
GCS   gs://evo-reward-ckpts  3 ckpt(s), latest step 75,000 (12:34:52 ago)

Training
  step           75,000/10,240,000  ( 0.73%)
  rate         34.9 steps/s   ETA: 3d 08h
  elapsed    00:35:46 (current invocation)
  gpu        100% util, 17148 MiB

Population
  prey  301   pred  50   food  600   E=187.2

Reward weights (prey)
  w_pred   -0.020 ± 0.660   ← fear emerging (want <0)
  w_prey   +0.020 ± 0.560   ← social affiliation (want >0)
```

Key signals to read at a glance:

- **VM status** — should be `RUNNING`. `TERMINATED` means preempted; the
  orchestrator will restart it, but you'd see that in the logs. If it
  says `missing`, the VM was deleted and a new one is being provisioned.
- **GCS checkpoints** — any count > 0 means training has made real
  progress that would survive a VM delete+recreate.
- **ETA** — refreshed each cycle from the current rate. Rough but
  honest. Gets more accurate as more samples accumulate.
- **w_pred / w_prey** — the Phase 1a science signal. You want `w_pred`
  drifting negative (fear of predators) and `w_prey` drifting positive
  (social affiliation of prey). The arrows turn green when the
  direction is right.
- **Retries** — the retry count in the "Retries" section shows
  Python-level crashes absorbed by the tmux loop. A small number
  (0-3) during a multi-day run is normal; many suggests a bug.

### Extra flags

- `--log N` — append the last `N` lines of `phase1a.log` to the output.
  Useful when the summary doesn't give you enough context.
- `--follow 30` — clears the screen and re-renders every 30 seconds.
  Run this in a side tmux pane while working; nothing else needed.

---

## Lifecycle operations

### Starting fresh

```bash
source scripts/gcloud-env.sh
caffeinate -d -i python3 scripts/spot_orchestrator.py --seed 0
```

The orchestrator handles everything: polling for spot capacity, creating
NAT/router in the chosen region, installing the VM env via the startup
script, uploading the repo, kicking off tmux. It runs forever in a
reconcile loop until you Ctrl-C.

### Resuming after closing your laptop

If you shut down / lost connection while a run was in progress:

```bash
source scripts/gcloud-env.sh
caffeinate -d -i python3 scripts/spot_orchestrator.py --seed 0
```

Same command. The orchestrator reads `.claude-orch-state.json` to find
the existing VM, doesn't create a new one, just resumes its reconcile
loop. Training on the VM kept running the whole time (it's in tmux),
or got automatically resumed by the VM's own startup script if the VM
was preempted and restarted during your absence.

### Switching from spot to on-demand

When spot is chronically stocked out and you'd rather pay the premium
for certainty (~3× cost, no preemption):

```bash
pkill -f spot_orchestrator       # stop the current monitor
rm -f .claude-orch-state.json    # optional — clears the zone pin
caffeinate -d -i python3 scripts/spot_orchestrator.py --seed 0 --on-demand
```

Any prior GCS checkpoints are picked up automatically on the new VM —
training resumes rather than starting from step 0.

### Tearing down when done

When Phase 1a has completed (`validate_replication.py` shows PASS),
clean up the infrastructure:

```bash
python3 scripts/status.py           # note the zone
source scripts/gcloud-env.sh

# Pull final results to your laptop
gcloud storage cp -r gs://evo-reward-ckpts/results ./gcp-results

# Delete the VM (preserves disk? No, this deletes everything)
gcloud compute instances delete evo-reward-gpu --zone=<zone> --quiet

# Delete NAT/routers (they bill ~$1/day each while alive)
gcloud compute routers nats delete evo-reward-nat --router=evo-reward-router \
  --region=us-central1 --quiet
gcloud compute routers delete evo-reward-router --region=us-central1 --quiet

# Optionally delete the GCS bucket too (if you've pulled everything)
gcloud storage rm -r gs://evo-reward-ckpts
```

---

## Troubleshooting

### `status.py` shows VM `missing`

The state file points to a zone, but the VM isn't there anymore. Either:

- You deleted it manually → orchestrator will re-poll on next
  reconcile cycle.
- GCP fully reclaimed a preempted spot VM (rare but happens). Same
  response: re-run orchestrator; it'll create a new VM.

### Log shows "0 bytes" but VM is RUNNING

Python stdout is block-buffered when piped through `tee`. We fixed
this with `stdbuf -oL -eL python -u` in the generated
`phase1a_loop.sh`. If you see it again, it means someone regenerated
the script without the buffering flags — regenerate via the
orchestrator's `upload_repo()` path, or manually edit
`~/phase1a_loop.sh` on the VM.

### Orchestrator is stuck at "all zones stockout"

Spot L4 capacity is genuinely absent across all 9 zones the
orchestrator rotates through. Options: (a) wait and re-check in 10-30
minutes — capacity rotates; (b) switch to `--on-demand` if you need
certainty.

### "all zones stockout" but on-demand also fails — check SSD quota

If the orchestrator reports stockout across all zones even with
`--on-demand`, that's almost certainly not a capacity issue. The
orchestrator logs an `↳ <error summary>` line under each failed
zone attempt — look for `SSD quota exceeded`. Each `g2-standard-8`
+ 200GB boot disk counts 200GB against the per-region
`SSD_TOTAL_GB` quota (default 250GB), so one orphaned disk from a
previously deleted VM is enough to block every new create in that
region.

```bash
# Inventory disks and look for ones not attached to a current VM:
gcloud compute disks list --format='table(name,zone.basename(),sizeGb,status,users.basename())'

# Delete any orphan (no USERS value):
gcloud compute disks delete <name> --zone=<zone> --quiet
```

### Preemption right after provision

Spot VMs are especially vulnerable in the first 15 min of life.
The orchestrator's startup script + GCS checkpoints make this
recoverable — but the orchestrator has no checkpoints to restore on
the very first attempt, so a cycle of "get VM → provision → preempt
→ re-poll" can happen multiple times. Each cycle is ~10 min and
costs < $0.15. If it happens 5+ times in a row, that's a signal spot
capacity is contested and you might want `--on-demand`.

### Telling if spot isn't going to work

If any of the following happen, give up on spot and switch to
`--on-demand`:

- **Preemption within 15 min** of first VM creation, repeatedly. In
  a healthy spot market, VMs typically live hours; repeated very-early
  preemption means capacity is contested.
- **No checkpoint has landed in GCS** after ≥2 preemption cycles
  (visible via `python3 scripts/status.py`: "no checkpoints yet" even
  after the VM has been up and retrying for 30+ min). That means each
  preemption wipes all progress, and retrying spot just spends
  provisioning time with no forward motion.
- **Orchestrator log shows the same zone repeatedly rotating through
  stockouts** even after the persistent-stockout fallback (delete+
  recreate in a new zone) kicks in. That's GCP-wide L4 demand
  pressure, not your problem.

Switching is a one-liner:

```bash
pkill -f spot_orchestrator
caffeinate -d -i python3 scripts/spot_orchestrator.py --seed 0 --on-demand
```

GCS-backed checkpoints from the spot attempts carry over to the
on-demand VM, so training resumes from wherever the last spot attempt
got to (if anywhere).

### VM shows RUNNING but `gcloud ssh` hangs

Usually means the IAP SSH firewall rule or Cloud NAT got deleted. Check:

```bash
gcloud compute firewall-rules list | grep iap
gcloud compute routers list
```

Re-create them per
[docs/gcp-setup.md § 6](gcp-setup.md#6-set-up-cloud-nat-and-iap-ssh-firewall).

### Training rate is much lower than expected

Check [`scripts/profile_sim_step.py`](../scripts/profile_sim_step.py) for
a per-phase breakdown on the VM. At steady state a spot L4 should hit
~25-35 steps/sec with population near cap. Much lower means a GPU
setup issue (`nvidia-smi` should show ~100% util during active
training).
