#!/usr/bin/env bash
# One-time setup for the public replays bucket.
#
# Creates gs://evo-reward-replays-public with public-read + CORS so the
# Vercel dashboard can fetch index.json and frames.bin directly. The bucket
# holds trajectory artifacts only — never checkpoint .npz files. Anything
# here is publicly readable by anyone on the internet; treat it as such.
#
# Override BUCKET / PROJECT / REGION via env before running.

set -euo pipefail

BUCKET="${EVO_REWARD_REPLAYS_BUCKET:-evo-reward-replays-public}"
PROJECT="${GCLOUD_PROJECT:-evo-reward}"
REGION="${GCLOUD_REGION:-us-central1}"

echo "Creating gs://$BUCKET in project=$PROJECT region=$REGION …"
gcloud storage buckets create "gs://$BUCKET" \
    --project="$PROJECT" \
    --location="$REGION" \
    --uniform-bucket-level-access

echo "Granting public read on gs://$BUCKET …"
gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" \
    --member="allUsers" \
    --role="roles/storage.objectViewer"

echo "Configuring CORS for browser fetches …"
CORS_TMP="$(mktemp)"
cat > "$CORS_TMP" <<'JSON'
[
  {
    "origin": ["*"],
    "method": ["GET", "HEAD"],
    "responseHeader": ["Content-Type", "Cache-Control"],
    "maxAgeSeconds": 300
  }
]
JSON
gcloud storage buckets update "gs://$BUCKET" --cors-file="$CORS_TMP"
rm -f "$CORS_TMP"

cat <<EOF

Done. Verify by running:

    python scripts/replay.py list-remote --bucket $BUCKET

Then in Vercel:

    VITE_REPLAYS_BASE_URL=https://storage.googleapis.com/$BUCKET/

The training VM will upload to this bucket automatically when config has
replay_record_interval_steps set and the VM has write access to it.
Grant VM write access with:

    gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" \\
        --member="serviceAccount:<vm-sa>@$PROJECT.iam.gserviceaccount.com" \\
        --role="roles/storage.objectAdmin"
EOF
