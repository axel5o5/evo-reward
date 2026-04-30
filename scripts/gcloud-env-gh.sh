# Per-shell gcloud env for the rl-bio-sims-494715 project (gh2707@columbia.edu).
#
# Source this file in any terminal where you want gcloud / launch_sweep_vm.sh
# to operate on rl-bio-sims-494715, without touching your global gcloud config:
#
#   source scripts/gcloud-env-gh.sh
#
# Counterpart to scripts/gcloud-env.sh (db3792@columbia.edu / evo-reward).
# Either file can be sourced in a shell to switch which project the launch
# scripts target. launch_sweep_vm.sh reads CLOUDSDK_CORE_PROJECT and
# GCS_BUCKET as overrides, with fallbacks to the original evo-reward defaults.

export CLOUDSDK_CORE_ACCOUNT=gh2707@columbia.edu
export CLOUDSDK_CORE_PROJECT=rl-bio-sims-494715
export GCS_BUCKET=rl-bio-sims-494715-ckpts

# One-time bootstrap (browser flow, run once per machine):
#   gcloud auth application-default login
#   mv ~/.config/gcloud/application_default_credentials.json \
#      ~/.config/gcloud/adc-rl-bio-sims.json
#   source scripts/gcloud-env-gh.sh
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/gcloud/adc-rl-bio-sims.json"
