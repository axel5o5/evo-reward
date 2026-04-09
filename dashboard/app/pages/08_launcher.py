"""
Launcher — Start experiments locally or via SSH.
Writes .pid files for the live monitor to track.
"""

import streamlit as st
import subprocess
import os
import signal
import yaml
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config_validator import validate_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
RESULTS_DIR = PROJECT_ROOT / "results"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"

st.set_page_config(page_title="Launcher", layout="wide")
st.title("Experiment Launcher")

# --- Mode selection ---
has_env = ENV_FILE.exists()
mode = st.sidebar.radio("Launch mode", ["Local", "Remote (SSH)"],
                         index=0, disabled=not has_env)

if mode == "Remote (SSH)" and not has_env:
    st.sidebar.warning("Create `.env` from `.env.example` for remote mode")
    mode = "Local"

# --- Config selection ---
config_files = sorted(CONFIGS_DIR.glob("*.yaml")) if CONFIGS_DIR.exists() else []
if not config_files:
    st.warning("No configs found in configs/")
    st.stop()

selected_config = st.selectbox("Config file", config_files,
                                format_func=lambda p: p.stem)

with open(selected_config) as f:
    config = yaml.safe_load(f)

# Validation warnings
issues = validate_config(config)
errors = [i for i in issues if i["severity"] == "error"]
if errors:
    st.error(f"{len(errors)} validation error(s) in config:")
    for e in errors:
        st.markdown(f"- **{e['key']}**: {e['issue']}")

# --- Parameters ---
col1, col2, col3 = st.columns(3)
with col1:
    seed = st.number_input("Seed", min_value=0, max_value=999, value=config.get("seed", 0))
with col2:
    max_steps = st.number_input("Max steps (0 = full run)", min_value=0,
                                 value=0, step=10000,
                                 help="Set to e.g. 10000 for a quick smoke test")
with col3:
    condition_name = config.get("experiment_name", selected_config.stem)
    st.text_input("Condition name", value=condition_name, disabled=True)

# Check if run_experiment.py exists
run_script = SCRIPTS_DIR / "run_experiment.py"
if not run_script.exists():
    st.warning(f"`{run_script.relative_to(PROJECT_ROOT)}` not found. "
               "Launcher requires the experiment entry point.")

# Check if already running
results_dir = RESULTS_DIR / condition_name / f"seed_{seed}"
pid_file = results_dir / ".pid"
if pid_file.exists():
    try:
        existing_pid = int(pid_file.read_text().strip())
        # Check if process is still running
        os.kill(existing_pid, 0)
        st.warning(f"Experiment already running (PID {existing_pid}). "
                   "Stop it first or choose a different seed.")
        if st.button("Force stop existing run"):
            try:
                os.kill(existing_pid, signal.SIGTERM)
                pid_file.unlink()
                st.success(f"Sent SIGTERM to PID {existing_pid}")
                st.rerun()
            except ProcessLookupError:
                pid_file.unlink()
                st.info("Process already dead, cleaned up PID file")
                st.rerun()
    except (ProcessLookupError, ValueError):
        # Process is dead, clean up stale PID file
        pid_file.unlink()

# --- Local Launch ---
if mode == "Local":
    st.subheader("Local Launch")

    cmd = [
        sys.executable, str(run_script),
        "--config", str(selected_config),
        "--seed", str(seed),
    ]
    if max_steps > 0:
        cmd.extend(["--max-steps", str(max_steps)])

    st.code(" ".join(cmd), language="bash")

    if st.button("Launch experiment", type="primary", disabled=not run_script.exists()):
        results_dir.mkdir(parents=True, exist_ok=True)

        # Copy config to results dir for provenance
        with open(results_dir / "config.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        # Launch subprocess
        log_file = results_dir / "stdout.log"
        with open(log_file, "w") as log_f:
            proc = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                cwd=str(PROJECT_ROOT),
            )

        # Write PID file
        pid_file.write_text(str(proc.pid))

        st.success(f"Launched! PID={proc.pid}, logging to {log_file.relative_to(PROJECT_ROOT)}")
        st.info("Go to **Live Monitor** to track progress.")

# --- Remote Launch ---
elif mode == "Remote (SSH)":
    st.subheader("Remote Launch (SSH + tmux)")

    # Load .env
    env_vars = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env_vars[k.strip()] = v.strip()

    remote_host = st.text_input("Remote host", value=env_vars.get("REMOTE_HOST", ""))
    remote_key = st.text_input("SSH key path", value=env_vars.get("REMOTE_KEY_PATH", "~/.ssh/id_rsa"))
    remote_path = st.text_input("Remote project path",
                                 value=env_vars.get("REMOTE_PROJECT_PATH", "~/evo-reward"))

    session_name = f"evo_{condition_name}_s{seed}"
    remote_cmd = (
        f"cd {remote_path} && "
        f"python scripts/run_experiment.py "
        f"--config configs/{selected_config.name} "
        f"--seed {seed}"
    )
    if max_steps > 0:
        remote_cmd += f" --max-steps {max_steps}"

    tmux_cmd = f"tmux new-session -d -s {session_name} '{remote_cmd}'"

    ssh_full = f"ssh -i {remote_key} {remote_host} \"{tmux_cmd}\""
    st.code(ssh_full, language="bash")

    if st.button("Launch remote", type="primary"):
        if not remote_host:
            st.error("Remote host is required")
        else:
            # First, scp config to remote
            scp_cmd = f"scp -i {remote_key} {selected_config} {remote_host}:{remote_path}/configs/"
            st.text(f"Copying config: {scp_cmd}")

            try:
                subprocess.run(scp_cmd, shell=True, check=True, capture_output=True, timeout=30)
                # Then launch via SSH
                result = subprocess.run(ssh_full, shell=True, capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    st.success(f"Launched in tmux session '{session_name}' on {remote_host}")
                    st.info(f"Attach with: `ssh {remote_host} -t 'tmux attach -t {session_name}'`")
                else:
                    st.error(f"SSH error: {result.stderr}")
            except subprocess.TimeoutExpired:
                st.error("SSH connection timed out")
            except Exception as e:
                st.error(f"Launch failed: {e}")

# --- Running experiments ---
st.divider()
st.subheader("Currently Running")

running = []
if RESULTS_DIR.exists():
    for pf in sorted(RESULTS_DIR.glob("*/seed_*/.pid")):
        try:
            pid = int(pf.read_text().strip())
            os.kill(pid, 0)  # Check if alive
            cond = pf.parent.parent.name
            seed_n = pf.parent.name
            running.append({"Condition": cond, "Seed": seed_n, "PID": pid})
        except (ProcessLookupError, ValueError):
            pass  # Dead process, will be cleaned by monitor

if running:
    st.dataframe(running, use_container_width=True)
else:
    st.info("No running experiments detected")
