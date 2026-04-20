# TODO: Replace tar-upload with `git clone` for VM deploys

**Status:** Not started. Deferred from 2026-04-20 — Phase 1a run in progress, too risky to refactor deploy path mid-run.

## Why

Currently `scripts/spot_orchestrator.py::upload_repo` tar-uploads the repo to the VM. That means `~/evo-reward` on the VM has no `.git` directory, so `git pull` fails. One-file patches require `gcloud compute scp`, which is fine for emergencies but awkward as a repeat workflow.

With a git-backed checkout we get:
- `git pull` on the VM for any code update
- Clean audit trail of exactly which commit the VM is running
- No need for the orchestrator to package + ship a tar on every deploy

## Auth constraint

Repo is private (`https://github.com/axel5o5/evo-reward`). VM needs credentials to clone. Options:

1. **Deploy key (recommended).** Generate an SSH keypair, add the public key to the repo's Settings → Deploy keys (leave "Allow write access" unchecked). Store the private key on the VM at `~/.ssh/id_ed25519` with `chmod 600`. Clone via SSH URL (`git@github.com:axel5o5/evo-reward.git`). Scoped to one repo, read-only, revocable per-VM.
2. **Personal access token.** Set up via `.netrc` or `git credential store`. Simpler to deploy, but the token has user-level scope — if the VM is compromised, attacker gets more than read access to one repo.
3. **Public repo.** Zero auth, trivial. Only viable if you're OK with the code being public.

Pick deploy key unless there's a reason not to.

## Scope

- **Do not modify the running VM.** Retrofitting `.git` into `~/evo-reward` on an in-flight training run is unnecessary risk. scp patches are fine for anything that comes up during the current run.
- **Design for the next VM.** The change only needs to affect future provisionings.

## Changes

### 1. `scripts/spot_orchestrator.py`

Replace or sibling the `upload_repo` function. New function signature:

```python
def clone_or_pull_repo(zone: str, seed: int, config: str, runtime: str,
                      branch: str = "dev/extensions") -> bool:
    """
    If ~/evo-reward/.git doesn't exist on the VM, clone fresh.
    Otherwise, fetch + reset --hard origin/<branch>.
    Preserves ~/evo-reward/results/.
    Returns True on success.
    """
```

Implementation outline:

```bash
# On the VM (run over gcloud compute ssh):
if [ -d ~/evo-reward/.git ]; then
  cd ~/evo-reward
  git fetch origin
  git checkout <branch>
  git reset --hard origin/<branch>
else
  # Preserve results/ if it exists
  mv ~/evo-reward/results /tmp/evo-reward-results 2>/dev/null || true
  rm -rf ~/evo-reward
  GIT_SSH_COMMAND="ssh -i ~/.ssh/evo_reward_deploy -o StrictHostKeyChecking=accept-new" \
    git clone git@github.com:axel5o5/evo-reward.git ~/evo-reward
  cd ~/evo-reward
  git checkout <branch>
  mv /tmp/evo-reward-results ~/evo-reward/results 2>/dev/null || true
fi
```

Then rewrite `phase1a_loop.sh` on the VM (unchanged flow) and restart the tmux session.

### 2. `scripts/vm_startup.sh`

Currently checks for `~/evo-reward` as a signal to install deps + start training. With git-clone, startup also needs to provision the deploy key before cloning:

- Pre-stage `~/.ssh/evo_reward_deploy` (the private key) via the orchestrator's SCP step (once, before the first clone).
- Add the `GIT_SSH_COMMAND` env into the clone invocation.
- Keep the rest of `vm_startup.sh` the same.

### 3. Deploy key setup (one-time, manually)

1. Generate keypair locally: `ssh-keygen -t ed25519 -f ~/.ssh/evo_reward_deploy -N ""`
2. Add `~/.ssh/evo_reward_deploy.pub` contents to GitHub → repo Settings → Deploy keys. Name it `evo-reward-gcp-vm`. Do **not** grant write access.
3. Save the private key somewhere the orchestrator can find it (e.g., `~/.config/evo-reward/deploy_key`). Do not commit.
4. Update `spot_orchestrator.py` to SCP this file onto the VM as part of provisioning.

### 4. Documentation

- Update `docs/gcp-setup.md` with the deploy-key setup steps.
- Remove (or mark deprecated) the tar-upload description.
- Add a note to `AGENTS.md` that VM code updates happen via `git pull` on the VM, not file scp.

## Migration path for the current running VM (optional, post-Phase-1a)

Once Phase 1a completes and we're ready to do Phase 1b on a fresh VM, that VM will naturally use the new git-clone path. No retrofit needed for the current VM — we'll just let it die when Phase 1a finishes.

## Acceptance criteria

- New VM provisioning runs `git clone` (not tar-upload) from a deploy-key-authenticated SSH URL.
- `gcloud compute ssh ... --command="cd ~/evo-reward && git pull"` works against the VM.
- `results/` directory is preserved across `git reset --hard` on the VM.
- Deploy key private file is not in the repo and is `chmod 600` on the VM.

## Estimated effort

~30 min coding + ~15 min setup (generate key, add to GitHub, test clone on a scratch VM).
