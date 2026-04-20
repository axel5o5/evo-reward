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

export CLOUDSDK_CORE_ACCOUNT=db3792@columbia.edu
export CLOUDSDK_CORE_PROJECT=evo-reward

# Quick verification — run `gcloud config list` after sourcing to confirm
# account + project show the evo-reward values above.
