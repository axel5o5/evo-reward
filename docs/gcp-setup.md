# GCP VM Setup for Phase 1a

Runbook for provisioning a GCP Compute Engine VM with an L4 GPU and running
Phase 1a (K&D baseline replication) end-to-end. Designed to be followed
top-to-bottom on a fresh GCP project.

Target machine: `g2-standard-8` + 1× NVIDIA L4 in `us-central1`.
Approximate cost: **~$0.70/hr running**, ~$0.02/hr stopped (disk only).
Phase 1a runtime: **~10–12 hours** per seed on L4.

This runbook uses **Cloud NAT + IAP tunneling** as the default. That means
the VM gets no public IP, egresses through a NAT gateway, and SSH goes
through an authenticated tunnel. It's more secure than the default path
and — importantly — it works under organization policies that forbid
external IPs on VMs (common on university / enterprise GCP projects).

---

## 1. Prerequisites

Install `gcloud` CLI (macOS):

```
brew install --cask google-cloud-sdk
```

Or follow https://cloud.google.com/sdk/docs/install for other platforms.

---

## 2. Create a per-project gcloud configuration

If you use gcloud for multiple projects, keep evo-reward isolated in its
own configuration so account and project settings don't clash.

```
gcloud config configurations create evo-reward
gcloud auth login <your-email>
gcloud config set account <your-email>
gcloud config set project evo-reward
```

Verify:

```
gcloud config configurations list
```

The `evo-reward` row should have `IS_ACTIVE = True`.

Switch between configs anytime with:

```
gcloud config configurations activate <other-config-name>
```

---

## 3. Enable required APIs

```
gcloud services enable compute.googleapis.com iap.googleapis.com \
  --project=evo-reward
```

Confirm:

```
gcloud services list --enabled --project=evo-reward | grep -E "compute|iap"
```

---

## 4. Check GPU quotas

Two quotas gate GPU VM creation:

- **Regional** `NVIDIA_L4_GPUS` in `us-central1` — usually 1 by default
- **Global** `GPUS_ALL_REGIONS` — **usually 0** on new projects, must be bumped

Check regional L4 quota:

```
gcloud compute regions describe us-central1 --project=evo-reward \
  --format="value(quotas)" | tr ';' '\n' | grep NVIDIA_L4_GPUS
```

Expected: `{'limit': 1.0, 'metric': 'NVIDIA_L4_GPUS', ...}`.

Check global all-regions GPU quota:

```
gcloud compute project-info describe --project=evo-reward \
  --format="value(quotas)" | tr ';' '\n' | grep GPUS_ALL_REGIONS
```

If limit is `0.0`, request an increase (next step). If ≥ 1, skip to step 6.

---

## 5. Request a quota increase (if needed)

There is no CLI path — this must be done via the console.

1. Open: https://console.cloud.google.com/iam-admin/quotas?project=evo-reward
2. In the Quota filter, search by metric name:
   `compute.googleapis.com/gpus_all_regions`
   (The human-readable filter "GPUs (all regions)" sometimes fails to
   match — use the metric name as a fallback.)
3. Check the row, click **EDIT QUOTAS**
4. New limit: `1`
5. Justification: *"Academic research — running reinforcement learning
   experiments on a single L4 GPU for a paper replication study."*
6. Submit

**Timing:** new paid accounts often get auto-approved in minutes;
free-trial accounts can take 1–2 business days and are sometimes denied.
If denied, see [Appendix A](#appendix-a-non-gcp-alternatives).

After approval, re-run the quota check. Propagation may take up to 15 min.

---

## 6. Set up Cloud NAT and IAP SSH firewall

Since the VM will have no public IP, it needs a NAT gateway to reach
pip / GitHub, and an IAP firewall rule so you can SSH in.

**Cloud Router** (required by Cloud NAT):

```
gcloud compute routers create evo-reward-router \
  --project=evo-reward \
  --region=us-central1 \
  --network=default
```

**Cloud NAT** (gives VMs without public IPs outbound internet access):

```
gcloud compute routers nats create evo-reward-nat \
  --project=evo-reward \
  --router=evo-reward-router \
  --region=us-central1 \
  --nat-all-subnet-ip-ranges \
  --auto-allocate-nat-external-ips
```

**Firewall rule for IAP SSH** (IAP's CIDR is `35.235.240.0/20`):

```
gcloud compute firewall-rules create allow-iap-ssh \
  --project=evo-reward \
  --network=default \
  --direction=INGRESS \
  --action=ALLOW \
  --rules=tcp:22 \
  --source-ranges=35.235.240.0/20
```

Cost note: Cloud NAT idles at ~$0.044/hr per gateway (~$1/day) even when
nothing is running. If you're done with the project, delete the NAT and
router after you delete the VM (see step 12).

---

## 7. Create the VM

GCP rotates Deep Learning VM image families periodically. Before running
the create command, list currently available CUDA 12.x images:

```
gcloud compute images list \
  --project=deeplearning-platform-release \
  --filter="family~'common-cu'" \
  --format="table(family,name,status)"
```

Pick the newest `ubuntu-2204` CUDA 12.x image family and substitute it
into `--image-family` below. As of April 2026 that's
`common-cu129-ubuntu-2204-nvidia-580` (CUDA 12.9).

```
gcloud compute instances create evo-reward-gpu \
  --project=evo-reward \
  --zone=us-central1-b \
  --machine-type=g2-standard-8 \
  --accelerator=type=nvidia-l4,count=1 \
  --image-family=common-cu129-ubuntu-2204-nvidia-580 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB \
  --maintenance-policy=TERMINATE \
  --metadata="install-nvidia-driver=True" \
  --no-address
```

Notes:

- `--no-address` is essential when the project has an org policy
  forbidding external IPs (e.g. `constraints/compute.vmExternalIpAccess`).
  It also works fine even without such a policy — SSH is via IAP.
- `jax[cuda12]` wheels are forward-compatible across CUDA 12.x minor
  versions, so any `common-cu12*` image works.
- `--maintenance-policy=TERMINATE` is required for GPU instances.
- `install-nvidia-driver=True` runs the NVIDIA driver installer on first
  boot; it may take 2–3 min after SSH becomes available.
- 200GB disk avoids the sub-200GB I/O performance warning.

**Zone stockout fallback:** L4 availability rotates across zones. If you
get `ZONE_RESOURCE_POOL_EXHAUSTED` on one zone, retry with another:
`us-central1-a`, `us-central1-b`, `us-central1-c`. Make sure all
downstream commands (`scp`, `ssh`, `stop`, etc.) use the matching zone.

---

## 8. SSH via IAP and install base tools

```
gcloud compute ssh evo-reward-gpu \
  --zone=us-central1-b --project=evo-reward \
  --tunnel-through-iap
```

First connection will generate an SSH key under `~/.ssh/google_compute_engine`
and may fail once with `failed to connect to backend` while SSH daemon
finishes booting — retry after 20 seconds.

On the VM, confirm the GPU is visible:

```
nvidia-smi
```

You should see one L4 listed with ~23GB VRAM. If `nvidia-smi` errors,
wait 1–2 min (driver install may still be running) and retry.

The Deep Learning VM image is minimal — no pip, git, or tmux
preinstalled. Install base packages and create a venv:

```
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv git tmux
python3 -m venv ~/evo-env
source ~/evo-env/bin/activate
pip install -U pip
```

From here on, every SSH session should start with
`source ~/evo-env/bin/activate` to get the venv on PATH.

Install JAX with CUDA support:

```
pip install -U "jax[cuda12]"
```

Verify JAX sees the GPU:

```
python -c "import jax; print(jax.devices())"
```

Expected: `[CudaDevice(id=0)]`.

---

## 9. Upload the repo

The evo-reward repo is private, so `git clone` over HTTPS won't work
without a token. Easiest path: tar your local clone and scp it over the
IAP tunnel.

From your **local** machine:

```
cd <parent-dir-of-your-evo-reward-clone>
tar --exclude='evo-reward/.git' \
    --exclude='evo-reward/__pycache__' \
    --exclude='evo-reward/**/__pycache__' \
    --exclude='evo-reward/results' \
    --exclude='evo-reward/dashboard/site/node_modules' \
    --exclude='evo-reward/dashboard/site/dist' \
    --exclude='evo-reward/.venv' \
    --exclude='evo-reward/.pytest_cache' \
    -czf /tmp/evo-reward.tar.gz evo-reward

gcloud compute scp /tmp/evo-reward.tar.gz \
  evo-reward-gpu:~/evo-reward.tar.gz \
  --zone=us-central1-b --project=evo-reward \
  --tunnel-through-iap
```

Archive is ~11MB. On the VM:

```
tar -xzf ~/evo-reward.tar.gz
rm ~/evo-reward.tar.gz
cd ~/evo-reward
source ~/evo-env/bin/activate
pip install -r requirements.txt
```

If tar prints warnings like `Ignoring unknown extended header keyword
'LIBARCHIVE.xattr.com.apple.provenance'`, they're harmless macOS
extended attributes — extraction still completes.

**Alternative for public repos:** just `git clone` on the VM.

Smoke-test with the full suite:

```
pytest -x -q
```

Expect `~104 passed, ~3 skipped` in about 4 minutes. If the test count
diverges, sync with the current main branch.

---

## 10. Run Phase 1a

Phase 1a is a single 10.24M-step run per seed, ~10–12h on L4. Run inside
`tmux` so it survives an SSH disconnect:

```
tmux new -s phase1a
source ~/evo-env/bin/activate
cd ~/evo-reward
python scripts/run_experiment_jax.py \
  --config configs/baseline_faithful.yaml \
  --runtime configs/runtime/gcp_l4.yaml \
  --seed 0
```

Detach with `Ctrl-b d`. Reattach with `tmux attach -t phase1a`.

To run multiple seeds sequentially (recommended: ≥3 for validation;
paper target is 5):

```
for seed in 0 1 2 3 4; do
  python scripts/run_experiment_jax.py \
    --config configs/baseline_faithful.yaml \
    --runtime configs/runtime/gcp_l4.yaml \
    --seed $seed
done
```

Total wall-clock for 5 seeds: ~55–60h. Cost: ~$40 compute + $1/day NAT.

Results land in `~/evo-reward/results/baseline_faithful/seed_<N>/`.

---

## 11. Validate results and download

After at least one seed finishes, on the VM:

```
source ~/evo-env/bin/activate
cd ~/evo-reward
python scripts/validate_replication.py \
  --results results/baseline_faithful/seed_0/
```

After ≥3 seeds finish, validate across all:

```
python scripts/validate_replication.py \
  --results results/baseline_faithful/ --all-seeds
```

Success criteria (from
[technical-spec-kd-replication.md](technical-spec-kd-replication.md)):

- `w_pred < 0` (fear of predators) — ≥3 of 5 seeds
- `w_prey > 0` (social affiliation) — ≥3 of 5 seeds

Exit code `0` = all criteria pass.

Pull results back to your **local** machine before tearing down:

```
gcloud compute scp --recurse --zone=us-central1-b --project=evo-reward \
  --tunnel-through-iap \
  evo-reward-gpu:~/evo-reward/results ./results-gcp
```

---

## 12. Stop or delete the VM (and NAT)

**Stop** (keeps disk, cheap to resume — ~$0.02/hr for 200GB disk):

```
gcloud compute instances stop evo-reward-gpu \
  --zone=us-central1-b --project=evo-reward
```

**Delete** (destroys everything including disk):

```
gcloud compute instances delete evo-reward-gpu \
  --zone=us-central1-b --project=evo-reward
```

Restart a stopped VM:

```
gcloud compute instances start evo-reward-gpu \
  --zone=us-central1-b --project=evo-reward
```

If you're done with the project entirely, also delete the NAT and
router to stop the ~$1/day NAT charge:

```
gcloud compute routers nats delete evo-reward-nat \
  --router=evo-reward-router --region=us-central1 \
  --project=evo-reward
gcloud compute routers delete evo-reward-router \
  --region=us-central1 --project=evo-reward
```

---

## Appendix A: Non-GCP alternatives

If GCP quota is denied or you want to skip the approval process:

- **Lambda Labs** (https://lambdalabs.com) — L4 at ~$0.75/hr, A100 at
  ~$1.29/hr, no quota gate. 2-minute provisioning. Recommended fallback.
- **Vast.ai** (https://vast.ai) — spot-market GPUs, often cheaper but
  less predictable; good for non-time-sensitive runs.
- **Paperspace Gradient** — notebook-style, decent for interactive work.

All three expose a standard Ubuntu + CUDA environment, so steps 8–11 of
this runbook work the same once you SSH in. Skip steps 4–7 (no quotas,
no NAT, no IAP — they give you a public IP and a plain SSH command).

---

## Appendix B: Troubleshooting

**`Constraint constraints/compute.vmExternalIpAccess violated`** — your
project (or its org) forbids external IPs on VMs. Make sure you used
`--no-address` in the create command and set up Cloud NAT (step 6).

**`ZONE_RESOURCE_POOL_EXHAUSTED` on VM create** — the zone is out of L4
stock right now. Retry with another zone in `us-central1` (a, b, c).
Regional quota applies to the whole region, so any zone works.

**IAP SSH fails with `failed to connect to backend`** — firewall rule
not set up, IAP API not enabled, or SSH daemon not ready yet on fresh
boot. Verify step 6 firewall rule exists and step 3 has IAP enabled,
then retry. First successful SSH after a new VM often takes 2 attempts.

**`nvidia-smi` says "NVIDIA-SMI has failed"** — driver install still
running. Wait 2 min and retry. If it persists after 5 min,
`sudo reboot` via SSH.

**JAX sees only CPU** — you installed `jax` instead of `jax[cuda12]`,
or you're outside the venv. `source ~/evo-env/bin/activate` first,
then `pip install -U "jax[cuda12]"`.

**`pip: command not found`** — Deep Learning VM images ship minimal;
run step 8's `apt-get install` first. Always activate the venv.

**Out-of-memory during JIT compile** — L4 has 24GB VRAM; baseline fits
comfortably. If you hit OOM, check for stale processes:
`nvidia-smi` → kill any unexpected PIDs with `kill -9 <pid>`.

**Quota page filter returns no results** — try the literal metric name
(`compute.googleapis.com/gpus_all_regions`) instead of the friendly
label ("GPUs (all regions)"). The UI filter is case-sensitive and picky.

**`git clone` errors with `could not read Username`** — repo is private.
Use the tar + scp path in step 9, or set up a GitHub PAT on the VM.
