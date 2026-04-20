# GCP Monitor + Control Plane — setup

Two related pieces on the `/gcp` dashboard page:

- **Part 1: Monitor** — read-only cron that polls GCP every 10 min and
  writes `gcp-status.json`. Covered below.
- **Part 2: Control plane** — PIN-gated Stop/Start/Restart/Delete buttons
  that trigger GitHub workflows. Covered at the bottom (search for
  `# Part 2`).

Quickest path if you're setting up from scratch: **do Part 2's step 1
first** (it creates one service account + WIF pool that Part 1 also
uses), then come back here.

---

# Part 1 — Monitor

Wires up a GitHub Actions cron that polls the `evo-reward` project every
10 minutes, writes `gcp-status.json` to the `gcp-status` orphan branch,
and surfaces it on the dashboard's `/gcp` page. Parallels
[scripts/status.py](../../scripts/status.py) but runs from CI and targets
only the signals that don't need SSH.

Pieces: [scripts/gcp_monitor.py](../../scripts/gcp_monitor.py),
[scripts/gcp_monitor_config.yaml](../../scripts/gcp_monitor_config.yaml),
[.github/workflows/gcp-monitor.yml](../../.github/workflows/gcp-monitor.yml),
[dashboard/site/src/pages/GcpMonitor.tsx](../site/src/pages/GcpMonitor.tsx).

## What it reads

| Source | Signal |
|---|---|
| Compute Engine API | `evo-reward-gpu` status, zone, machine, SPOT vs STANDARD, runtime |
| Cloud Storage API | `gs://evo-reward-ckpts` — checkpoint count, latest step, last-saved age, bucket size |
| Config (static) | Hourly pricing, NAT rate, `nat_active_since` timestamp |
| BigQuery (optional) | Billing export for authoritative cost, ~24h lag |

## One-time setup

### 1. Enable APIs

```bash
source scripts/gcloud-env.sh
gcloud services enable \
  compute.googleapis.com \
  storage.googleapis.com \
  bigquery.googleapis.com \
  iamcredentials.googleapis.com
```

(The last one — `iamcredentials` — is needed by Workload Identity
Federation to mint impersonated tokens. Omitting it produces a
`SERVICE_DISABLED` error on every monitor run, visible as a yellow
banner on the dashboard.)

### 2. Give the worker read-only access

Pick one path.

**Path A — Workload Identity Federation (recommended, no keys to rotate):**

Create a service account in `evo-reward` and a WIF pool that trusts this
GitHub repo, then grant the SA these roles on `evo-reward`:

- `roles/compute.viewer`
- `roles/storage.objectViewer` on `gs://evo-reward-ckpts`
- `roles/bigquery.dataViewer` on the billing export dataset (only if using billing)
- `roles/bigquery.jobUser` on the project (only if using billing)

Add these as **repo variables** (Settings → Secrets and variables →
Actions → Variables tab):

```
GCP_WIF_PROVIDER         = projects/<NUM>/locations/global/workloadIdentityPools/gh/providers/gh-provider
GCP_WIF_SERVICE_ACCOUNT  = gcp-monitor@evo-reward.iam.gserviceaccount.com
```

**Path B — service account JSON key (simpler, one key to rotate):**

Create the SA, grant the same roles, download the key JSON, add as repo
**secret** `GCP_SA_KEY_DEFAULT`.

### 3. (Optional) Enable billing export for `billing_actual_usd`

Without this, live-estimate is the only number shown and that's fine.
To add authoritative spend:

1. In the Billing console, turn on **Detailed usage cost export to
   BigQuery**. Target a dataset (usually in a dedicated billing project).
2. In [scripts/gcp_monitor_config.yaml](../../scripts/gcp_monitor_config.yaml),
   set `billing.export_table` to the resulting
   `billing-proj.billing.gcp_billing_export_v1_XXXXXX` and
   `billing.account_id` to the billing account id.
3. Grant `roles/bigquery.dataViewer` on that dataset to the worker SA.

### 4. (Optional) Capture NAT "always-on" cost

Cloud NAT bills ~$0.044/hr even while the VM is off. To surface how much
it's accumulated since you ran gcp-setup.md step 6, set
`infra_costs.nat_active_since` in the config to that timestamp
(e.g. `"2026-04-10T00:00:00Z"`). Leave null to skip the NAT line entirely.

### 5. Point the dashboard at the data

**If the repo is public**, you can point the site directly at the raw URL:

```
VITE_GCP_STATUS_URL = https://raw.githubusercontent.com/<user>/evo-reward/gcp-status/gcp-status.json
```

**If the repo is private** (the default), leave `VITE_GCP_STATUS_URL` unset
and the site will fetch from `/api/status` — a Vercel serverless route
([api/status.ts](../../api/status.ts)) that reads the raw file with a
server-side PAT. See step 2 of Part 2 below for the PAT — the same token
covers both the control plane and this proxy, just add `Contents: Read`
to its scopes.

Redeploy either way (VITE_* vars are baked at build time).

### 6. First run

Actions → `GCP Monitor` → Run workflow. First run creates the
`gcp-status` branch and pushes the initial `gcp-status.json`. Open `/gcp`.

## Local development

```bash
# Synthetic payload; no GCP creds needed:
python3 scripts/gcp_monitor.py --dry-run --out dashboard/site/public/gcp-status.json

# Point dev server at the local file:
echo 'VITE_GCP_STATUS_URL=/gcp-status.json' > dashboard/site/.env.local
cd dashboard/site && npm run dev
```

To test against real GCP from your Mac:

```bash
source scripts/gcloud-env.sh    # sets CLOUDSDK_CORE_PROJECT=evo-reward
python3 scripts/gcp_monitor.py --out /tmp/gcp-status.json
cat /tmp/gcp-status.json | jq
```

## Alignment with status.py

`scripts/status.py` is the rich local view — it SSHes into the VM, pulls
`~/phase1a.log`, and parses the 8-weight evolution signal. The dashboard
monitor intentionally does **less**: it doesn't SSH, so it's safe to run
from CI without tunnel credentials. Both share the same VM and GCS probes
so their numbers should agree.

To surface training progress (step, sps, reward weights) on the dashboard,
see "Future: training progress" below.

## Missing on purpose

- **Batch API** — we don't use Google Batch. Dropped.
- **Multi-project / partner support** — simplified to single project.
  If a collaborator ever grants access, add another probe rather than
  rebuilding the multi-owner structure.

## Future: training progress on the dashboard

The `training` field in the payload is always `null` today. To populate
it without giving CI SSH credentials, the cleanest path is:

1. Have the training loop write a short `progress.json` alongside its
   checkpoints — step, sps, latest reward weights, last log timestamp.
2. The existing `gs://evo-reward-ckpts` rsync sidecar already pushes
   everything in `results/` to GCS every 5 min, so the file lands in
   the bucket for free.
3. Add one more probe in `gcp_monitor.py` (e.g. `probe_training()`) that
   reads the most-recent `progress.json` from the bucket and populates
   `payload["training"]`.

~20 lines of Python + a small emit helper in the runner. Say the word and
I'll scaffold it.

## Cost of the monitor itself

- Compute API `instances.get` across 9 zones: well under free tier.
- Storage API bucket listing: free for the first 5k class-A ops/month;
  we do ~4.3k/month (every 10 min) on a small bucket — borderline but
  still free.
- BigQuery billing query: ~1–10 MB per call against a partitioned table.
- GitHub Actions: ~30 s × 6/hr × 24 × 30 = ~72 min/month (free tier).

---

# Part 2 — Control plane (PIN-gated actions)

The `/gcp` page has a control panel for Stop, Start, Restart, Delete.
User enters a shared PIN, clicks a button, and the action runs on GCP
with a ~1 min feedback loop (next monitor poll).

Flow:

```
Browser → [PIN + Stop] → /api/action (Vercel function)
    ↓ PIN check → POST workflow_dispatch
GitHub Actions → gcp-action.yml → auth via WIF → gcloud compute ...
    ↓
Monitor polls GCS / Compute → gcp-status.json updates → dashboard re-renders
```

Pieces: [api/action.ts](../../api/action.ts),
[.github/workflows/gcp-action.yml](../../.github/workflows/gcp-action.yml),
`ControlPanel` in [GcpMonitor.tsx](../site/src/pages/GcpMonitor.tsx).

## One-time setup (~30 min)

### 1. Create the service account + WIF pool in GCP

Service account and roles:

```bash
source scripts/gcloud-env.sh

gcloud iam service-accounts create gcp-dashboard \
  --display-name="Dashboard controller"

# Enough to stop/start/delete the VM. Narrower than compute.admin.
gcloud projects add-iam-policy-binding evo-reward \
  --member="serviceAccount:gcp-dashboard@evo-reward.iam.gserviceaccount.com" \
  --role="roles/compute.instanceAdmin.v1"

# Read-only for the monitor workflow (reuses the same SA).
gcloud projects add-iam-policy-binding evo-reward \
  --member="serviceAccount:gcp-dashboard@evo-reward.iam.gserviceaccount.com" \
  --role="roles/storage.objectViewer"
```

WIF pool + provider:

```bash
gcloud iam workload-identity-pools create github-pool \
  --location=global \
  --display-name="GitHub Actions"

gcloud iam workload-identity-pools providers create-oidc github-provider \
  --location=global \
  --workload-identity-pool=github-pool \
  --display-name="GitHub OIDC" \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --attribute-condition="assertion.repository == 'YOUR_GH_USER/evo-reward'"
```

Bind the repo to the SA (replace `YOUR_GH_USER` and the project number):

```bash
# Get the project number (not the id):
PROJECT_NUMBER=$(gcloud projects describe evo-reward --format="value(projectNumber)")

gcloud iam service-accounts add-iam-policy-binding \
  gcp-dashboard@evo-reward.iam.gserviceaccount.com \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/github-pool/attribute.repository/YOUR_GH_USER/evo-reward"
```

Print the values you need for GitHub:

```bash
echo "GCP_WIF_PROVIDER        = projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/github-pool/providers/github-provider"
echo "GCP_WIF_SERVICE_ACCOUNT = gcp-dashboard@evo-reward.iam.gserviceaccount.com"
```

### 2. Configure GitHub

Repo → Settings → Secrets and variables → Actions → **Variables tab**
(these are non-secret, they show up in logs):

```
GCP_WIF_PROVIDER          = <value from step 1>
GCP_WIF_SERVICE_ACCOUNT   = gcp-dashboard@evo-reward.iam.gserviceaccount.com
```

Manually trigger the monitor once (Actions → GCP Monitor → Run workflow)
to verify WIF works before wiring up Vercel.

### 3. Mint a GitHub PAT for Vercel

The Vercel functions use this token to (a) trigger workflow_dispatch
via the GitHub API and (b) read `gcp-status.json` from the private
gcp-status branch. It needs a PAT scoped *only* to this repo.

1. https://github.com/settings/personal-access-tokens/new
2. Fine-grained token, resource owner = you (or the org).
3. Repository access: **only `evo-reward`**.
4. Permissions:
   - `Actions: Read and write` (for /api/action → workflow_dispatch)
   - `Contents: Read` (for /api/status → read gcp-status.json)
5. Expiration: 90 days (set a calendar reminder to rotate).
6. Copy the token (starts with `github_pat_...`).

### 4. Configure Vercel env vars

Vercel → Project → Settings → Environment Variables. Add for
**Production** (and Preview, if you want to test):

| Name | Value |
|---|---|
| `DASHBOARD_PIN` | a shared PIN (8+ chars; this is your whole auth) |
| `GH_DISPATCH_TOKEN` | the fine-grained PAT from step 3 |
| `GH_OWNER` | your GitHub username, e.g. `axel5o5` |
| `GH_REPO` | `evo-reward` |
| `GH_REF` | `main` (branch the workflow runs on) |

Redeploy the site after adding these — Vercel bakes env vars at deploy
time for functions.

### 5. Test

1. Open `/gcp`. You should see the monitor cards + a "Control panel"
   below them with a PIN input and 4 buttons.
2. Wrong PIN → red `invalid_pin` error, no API call leaks anywhere.
3. Right PIN + Stop → button goes to "…" for a second, then green
   "Dispatched stop · req abc123".
4. Open the Actions tab in GitHub. There should be a `GCP Action` run
   with `request_id=abc123` in its logs.
5. Within ~1 min the monitor poll picks up the new state and the VM
   card flips to TERMINATED.

## Failure modes & what they mean

| Symptom | Cause | Fix |
|---|---|---|
| Button shows `server_misconfigured` | Vercel env var missing | Re-check step 4, redeploy |
| `invalid_pin` despite correct PIN | Trailing whitespace in env var | Edit the Vercel value |
| `github_dispatch_failed (404)` | PAT can't see the workflow | Token scope wrong (step 3), or `GH_OWNER`/`GH_REPO` mismatch |
| Workflow fails with `permission denied` on gcloud | WIF binding missing | Re-run step 1 bind command |
| Start fails with stockout | Spot capacity gone | Delete instead; orchestrator respawns |

## Deferred to v2

- **Force GCS sync** and **Tail log**: need SSH-through-IAP in the
  workflow (extra `roles/iap.tunnelResourceAccessor` on the SA + firewall
  rule already exists via gcp-setup.md). Straightforward to add.
- **Switch SPOT ↔ on-demand**: requires a small refactor to
  `spot_orchestrator.py` so it reads its provisioning mode from a
  runtime flag (GCS file or repo variable) instead of a CLI arg at
  startup.
