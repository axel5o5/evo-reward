"""
Extract data from evo-reward source code, configs, and docs
into JSON files consumed by the static dashboard site.

Run manually:   python dashboard/extract_data.py
Auto-run:       Vite plugin calls this before dev/build

Outputs to: dashboard/site/src/data/
"""

import json
import re
import subprocess
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(__file__).resolve().parent / "site" / "src" / "data"


def extract_config_schema():
    """Parse baseline_faithful.yaml into a structured JSON with metadata."""
    config_path = PROJECT_ROOT / "configs" / "baseline_faithful.yaml"
    if not config_path.exists():
        return {}

    with open(config_path) as f:
        raw_lines = f.readlines()

    # Parse YAML values
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Extract comments as metadata
    entries = []
    current_section = ""
    current_comment = ""

    for line in raw_lines:
        stripped = line.strip()

        # Section headers (lines like "# --- Section ---")
        section_match = re.match(r"^#\s*[─—-]+\s*(.+?)\s*[─—-]+", stripped)
        if section_match:
            current_section = section_match.group(1).strip()
            current_comment = ""
            continue

        # Comment lines
        if stripped.startswith("#"):
            comment_text = stripped.lstrip("# ").strip()
            if comment_text:
                current_comment = (current_comment + " " + comment_text).strip()
            continue

        # Key-value lines
        kv_match = re.match(r"^(\w+):\s*(.+?)(?:\s*#\s*(.*))?$", stripped)
        if kv_match:
            key = kv_match.group(1)
            inline_comment = kv_match.group(3) or ""
            value = config.get(key)

            entries.append({
                "key": key,
                "value": value,
                "section": current_section,
                "comment": current_comment,
                "inline_comment": inline_comment.strip(),
            })
            current_comment = ""

        # Blank lines reset accumulated comment
        if not stripped:
            current_comment = ""

    return {"parameters": entries, "source": "configs/baseline_faithful.yaml"}


def extract_obs_layout():
    """Parse the observation vector layout from interfaces.md."""
    return {
        "total_dim": 205,
        "segments": [
            {
                "name": "proximity_sensors",
                "start": 0, "end": 127,
                "shape": [32, 4],
                "description": "32 sensors x 4 channels [prey, predator, food, wall]",
                "details": "Winner-take-all: only closest type positive; others = -1.0. "
                           "FOV: 120 deg, range: 200 units.",
                "range": "[-1, 1]",
            },
            {
                "name": "tactile_collision",
                "start": 128, "end": 199,
                "shape": [4, 18],
                "description": "4 type channels x 18 bins at 20 deg spacing",
                "details": "Channels: [conspecific, other_species, food, wall]. Binary contact.",
                "range": "{0, 1}",
            },
            {
                "name": "velocity",
                "start": 200, "end": 201,
                "shape": [2],
                "description": "2D velocity (vx, vy)",
                "range": "[-10, 10]",
            },
            {
                "name": "angle",
                "start": 202, "end": 202,
                "shape": [1],
                "description": "Agent heading in radians",
                "range": "[-2pi, 2pi]",
            },
            {
                "name": "angular_velocity",
                "start": 203, "end": 203,
                "shape": [1],
                "description": "Angular velocity (radians/step)",
                "range": "[-pi/10, pi/10]",
            },
            {
                "name": "energy",
                "start": 204, "end": 204,
                "shape": [1],
                "description": "Raw energy value",
                "range": "[0, 1000]",
            },
        ],
    }


def extract_module_deps():
    """Parse import statements from src/*.py to build a dependency graph."""
    src_dir = PROJECT_ROOT / "src"
    if not src_dir.exists():
        return {"nodes": [], "edges": []}

    modules = {}
    for py_file in sorted(src_dir.glob("*.py")):
        if py_file.name.startswith("__"):
            continue
        name = py_file.stem
        with open(py_file) as f:
            content = f.read()
        lines = len(content.splitlines())
        modules[name] = {"name": name, "lines": lines, "file": f"src/{py_file.name}"}

    edges = []
    for py_file in sorted(src_dir.glob("*.py")):
        if py_file.name.startswith("__"):
            continue
        name = py_file.stem
        with open(py_file) as f:
            content = f.read()

        for other_name in modules:
            if other_name == name:
                continue
            # Check for imports like "from src.reward import", "from .reward import",
            # "import reward", "from reward import"
            patterns = [
                rf"from\s+src\.{other_name}\s+import",
                rf"from\s+\.?{other_name}\s+import",
                rf"import\s+(?:src\.)?{other_name}",
            ]
            for pat in patterns:
                if re.search(pat, content):
                    edges.append({"source": name, "target": other_name})
                    break

    return {
        "nodes": list(modules.values()),
        "edges": edges,
    }


def extract_sim_loop():
    """The 10-step simulation loop with module links."""
    return {
        "steps": [
            {"number": 1, "action": "Get observations for all agents",
             "module": "agents.py", "function": "get_observation()"},
            {"number": 2, "action": "Sample actions, write to rollout buffer",
             "module": "policy.py", "function": "sample_action()"},
            {"number": 3, "action": "Step physics (phyjax2d)",
             "module": "environment.py", "function": "step_physics()"},
            {"number": 4, "action": "Check eating events",
             "module": "environment.py", "function": "check_eating()"},
            {"number": 5, "action": "Compute rewards, write to rollout",
             "module": "agents.py", "function": "compute_reward()"},
            {"number": 6, "action": "Update energies",
             "module": "lifecycle.py", "function": "update_energies()"},
            {"number": 7, "action": "Process births and deaths",
             "module": "lifecycle.py", "function": "process_births_and_deaths()"},
            {"number": 8, "action": "Regenerate food",
             "module": "lifecycle.py", "function": "regenerate_food()"},
            {"number": 9, "action": "PPO update (if rollout buffer full)",
             "module": "ppo.py", "function": "ppo_update()"},
            {"number": 10, "action": "Log metrics / save checkpoint",
             "module": "metrics.py", "function": "log_step()"},
        ]
    }


def extract_git_timeline():
    """Parse git log into a timeline of project milestones."""
    # Use record separator to reliably split multi-line bodies
    SEP = "---COMMIT---"
    fmt = f"%H|%s|%ai|%an|%b{SEP}"
    try:
        result = subprocess.run(
            ["git", "log", f"--format={fmt}", "--all", "--reverse"],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )
        if result.returncode != 0:
            return {"commits": []}
    except FileNotFoundError:
        return {"commits": []}

    commits = []
    for block in result.stdout.split(SEP):
        block = block.strip()
        if not block:
            continue
        # First line contains hash|subject|date|author|body-start
        # Body may span multiple lines
        parts = block.split("|", 4)
        if len(parts) < 4:
            continue
        hash_, message, date, author = parts[0], parts[1], parts[2], parts[3]
        body = parts[4].strip() if len(parts) > 4 else ""
        # Strip Co-Authored-By lines from body
        body_lines = [l for l in body.splitlines()
                      if not l.strip().startswith("Co-Authored-By:")]
        body = "\n".join(body_lines).strip()

        # Detect session/phase tags
        phase_match = re.match(r"\[Phase (\d+)\]", message)
        session_match = re.match(r"\[Session (\d+)\]", message)

        commits.append({
            "hash": hash_[:8],
            "message": message,
            "body": body or None,
            "date": date.strip(),
            "author": author.strip(),
            "phase": phase_match.group(1) if phase_match else None,
            "session": session_match.group(1) if session_match else None,
        })

    return {"commits": commits}


def extract_deviations():
    """Parse emevo-diff.md D-entries into structured data."""
    diff_path = PROJECT_ROOT / "docs" / "emevo-diff.md"
    if not diff_path.exists():
        return {"deviations": []}

    with open(diff_path) as f:
        content = f.read()

    deviations = []
    # Match patterns like ### [D1] Title or ### ~~[D5] Title~~
    pattern = r"### (~~)?\[D(\d+)\]\s*(.+?)(?:~~)?\n(.*?)(?=\n### |\n---|\Z)"
    for match in re.finditer(pattern, content, re.DOTALL):
        struck = match.group(1) is not None
        number = int(match.group(2))
        title = match.group(3).strip()
        body = match.group(4).strip()

        deviations.append({
            "id": f"D{number}",
            "title": title,
            "resolved": struck,
            "body_preview": body[:200] + "..." if len(body) > 200 else body,
        })

    return {"deviations": deviations}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    extractors = {
        "config-schema.json": extract_config_schema,
        "obs-layout.json": extract_obs_layout,
        "module-deps.json": extract_module_deps,
        "sim-loop-steps.json": extract_sim_loop,
        "git-timeline.json": extract_git_timeline,
        "deviations.json": extract_deviations,
    }

    for filename, extractor in extractors.items():
        data = extractor()
        output_path = OUTPUT_DIR / filename
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"  Extracted {filename} ({len(json.dumps(data))} bytes)")

    print(f"\nAll data extracted to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
