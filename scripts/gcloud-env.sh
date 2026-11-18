# Per-shell gcloud env for the evo-reward GCP project.
#
# Source this file in any terminal where you want gcloud to operate on
# evo-reward, without touching your global gcloud config or active
# account:
#
#   source scripts/gcloud-env.sh
#
# The env vars below override the globally-active gcloud config for this
# shell session only. Other terminals / other projects see their usual
# defaults. Unset with `unset CLOUDSDK_CORE_ACCOUNT CLOUDSDK_CORE_PROJECT`
# or just close the shell.

# Set EVO_GCP_ACCOUNT (your gcloud account) before sourcing; EVO_GCP_PROJECT and
# GCS_BUCKET default to the evo-reward project and its checkpoint bucket.
export CLOUDSDK_CORE_ACCOUNT="${EVO_GCP_ACCOUNT:?set EVO_GCP_ACCOUNT to your gcloud account first}"
export CLOUDSDK_CORE_PROJECT="${EVO_GCP_PROJECT:-evo-reward}"
export GCS_BUCKET="${GCS_BUCKET:-evo-reward-ckpts}"

# Per-project Application Default Credentials file. Python google-cloud
# libraries (used by scripts/gcp_monitor.py) read ADC from this env var if
# set, otherwise from a single shared file at
# ~/.config/gcloud/application_default_credentials.json — which gets
# overwritten by every `gcloud auth application-default login`, so two
# parallel projects on different accounts would stomp on each other.
# Pinning a per-project path here mirrors the CLOUDSDK_CORE_* scoping above.
#
# One-time bootstrap:
#   gcloud auth application-default login     # browser flow, uses current account
#   mv ~/.config/gcloud/application_default_credentials.json \
#      ~/.config/gcloud/adc-evo-reward.json
#   source scripts/gcloud-env.sh              # now Python libs use the project file
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/gcloud/adc-evo-reward.json"

# Quick verification — run `gcloud config list` after sourcing to confirm
# account + project show the evo-reward values above.
