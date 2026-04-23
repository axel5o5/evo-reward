"""replay_recorder.py
---------------------
In-line replay recorder — captures actual training trajectories into short
windows leading up to each flush boundary, then writes + uploads + prunes.

Usage from the runner:

    recorder = ReplayRecorder(config, exp_name, seed, local_out_root, bucket=...)
    for step in range(start_step, total_steps):
        sim_state = sim_step_core(sim_state)
        recorder.step(sim_state, step + 1)        # step+1 = post-step count

Key properties:
  * Zero cost outside the recording window (no host↔device sync); capture + copy
    only kicks in for the last `replay_record_length_steps` steps before each
    `replay_record_interval_steps` boundary.
  * Resume-safe: buffer starts empty on construction, re-fills from the resume
    point, flushes a shorter replay if resume lands mid-window.
  * Quantization: when `replay_quantize` is true, positions and food positions
    go to uint16 (scale = world_size/65535, ~0.015 unit precision) and energy
    goes to uint8 (scale = energy_capacity/255). ~2–3× smaller on disk.

Binary format is versioned via meta.json.version. v2 (current) adds per-frame
identity/lineage/phenotype tracks on top of v1. Dtypes below assume quantize=True:
  agent_ids      uint16  (n_frames, max_agents)     stored = (id - id_base + 1),
                                                    0 = sentinel (-1). id_base
                                                    lives in meta.id_base.
                                                    Falls back to int32 when
                                                    the window's id range
                                                    exceeds 65534.
  parent_ids     uint16  (n_frames, max_agents)     same encoding as agent_ids
  ages           int32   (n_frames, max_agents)     birth_step = step - age
  reward_weights int8    (n_frames, max_agents, 4)  scale = 4/127 (covers ±4)
  action         int8    (n_frames, max_agents, 2)  scale = 1/127 (covers ±1)
Per-frame (not static) because slot reuse after death replaces the occupant
mid-window — a static section would misattribute to the previous tenant.
Unquantized replays keep everything as float32 / int32.

v2 storage at defaults (length=1000, max_agents=500): ~14 MB (vs ~8 MB v1).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


class ReplayRecorder:
    def __init__(
        self,
        config: dict,
        exp_name: str,
        seed: int,
        local_out_root: str | os.PathLike,
        bucket: str | None = None,
        run_tag: str = "",
    ):
        self.interval = int(config.get("replay_record_interval_steps", 0) or 0)
        self.length = int(config.get("replay_record_length_steps", 1000) or 0)
        self.enabled = self.interval > 0 and self.length > 0
        if not self.enabled:
            return

        self.exp_name = exp_name
        self.seed = seed
        self.run_tag = run_tag
        self.max_agents = int(config["prey_cap"]) + int(config["predator_cap"])
        self.food_max = int(config["food_max"])
        self.world_size = float(config["world_size"])
        self.energy_capacity = float(config.get("energy_capacity", 1000.0))
        self.quantize = bool(config.get("replay_quantize", True))
        self.bucket = bucket
        self.retention_policy = str(config.get("replay_retention_policy", "last_n"))
        self.retention_config = dict(config.get("replay_retention_config", {"keep_last_n": 10}))

        # local_out_root is already `<out_dir>/<exp>/seed_<N>[/<run_tag>]/replays`
        # when called from the runner, so don't re-append exp/seed here.
        self.local_root = Path(local_out_root)
        self.local_root.mkdir(parents=True, exist_ok=True)

        self._alloc_buffer()

    # ---------------------------- buffer -------------------------------------

    def _alloc_buffer(self) -> None:
        L, N, F = self.length, self.max_agents, self.food_max
        self._buf = {
            "pos":         np.empty((L, N, 2), dtype=np.float32),
            "angle":       np.empty((L, N),    dtype=np.float32),
            "energy":      np.empty((L, N),    dtype=np.float32),
            "alive":       np.empty((L, N),    dtype=np.uint8),
            "food_pos":    np.empty((L, F, 2), dtype=np.float32),
            "food_active": np.empty((L, F),    dtype=np.uint8),
            "step":        np.empty((L,),      dtype=np.int32),
            # v2 phenotype/identity tracks. Per-frame because slot reuse after
            # death replaces the occupant mid-window; a "written once" static
            # section would silently misattribute parent/reward to the previous
            # tenant.
            "agent_ids":      np.empty((L, N),    dtype=np.int32),
            "parent_ids":     np.empty((L, N),    dtype=np.int32),
            "ages":           np.empty((L, N),    dtype=np.int32),
            "reward_weights": np.empty((L, N, 4), dtype=np.float32),
            "action":         np.empty((L, N, 2), dtype=np.float32),
        }
        self._count = 0

    # ---------------------------- step hook ----------------------------------

    def step(self, sim_state, step_after: int) -> None:
        """Call once per training tick after sim_step_core returns.

        `step_after` is the count of completed steps (use the Python loop var
        `step + 1`, not `sim_state.step` — that forces a host sync each call).
        """
        if not self.enabled:
            return

        # Upcoming boundary we're racing toward — handles the case where
        # step_after lands exactly on a boundary (which should flush).
        upcoming = ((step_after - 1) // self.interval + 1) * self.interval
        in_window = (upcoming - step_after) < self.length

        if in_window:
            self._capture(sim_state, step_after)
        if step_after == upcoming and self._count > 0:
            self._flush(sim_state)

    # ---------------------------- capture ------------------------------------

    def _capture(self, sim_state, step_after: int) -> None:
        if self._count >= self.length:
            return
        i = self._count
        circle = sim_state.phyjax_stated.get("circle")
        # np.asarray forces a sync. Fine — we're only in the window for `length`
        # out of `interval` steps (~1% at defaults).
        self._buf["pos"][i] = np.asarray(circle.p.xy)
        self._buf["angle"][i] = np.asarray(circle.p.angle)
        self._buf["energy"][i] = np.asarray(sim_state.energies)
        self._buf["alive"][i] = np.asarray(sim_state.is_active, dtype=np.uint8)
        self._buf["food_pos"][i] = np.asarray(sim_state.food_positions)
        self._buf["food_active"][i] = np.asarray(sim_state.food_active, dtype=np.uint8)
        self._buf["step"][i] = int(step_after)

        self._buf["agent_ids"][i] = np.asarray(sim_state.agent_ids, dtype=np.int32)
        self._buf["parent_ids"][i] = np.asarray(sim_state.parent_ids, dtype=np.int32)
        self._buf["ages"][i] = np.asarray(sim_state.ages, dtype=np.int32)
        self._buf["reward_weights"][i] = np.asarray(sim_state.reward_weights, dtype=np.float32)
        # Last-written action from the per-slot circular rollout buffer.
        # rollout_ptrs advances after each sim_step, so ptrs-1 (mod len) is
        # the action that drove this step. Right after a PPO update ptrs
        # resets to 0 and this falls back to the stale tail of the buffer —
        # acceptable artifact (affects at most 1 frame per PPO boundary).
        rollout_actions = np.asarray(sim_state.rollout_actions)
        rollout_ptrs = np.asarray(sim_state.rollout_ptrs)
        last = (rollout_ptrs - 1) % rollout_actions.shape[1]
        self._buf["action"][i] = rollout_actions[np.arange(self.max_agents), last]
        self._count += 1

    # ---------------------------- flush --------------------------------------

    def _flush(self, sim_state) -> None:
        n = self._count
        start_step = int(self._buf["step"][0])
        out_dir = self.local_root / f"step_{start_step:08d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        species = np.asarray(sim_state.species, dtype=np.int32)
        radii = np.asarray(sim_state.radii, dtype=np.float32)

        pos = self._buf["pos"][:n]
        angle = self._buf["angle"][:n]
        energy = self._buf["energy"][:n]
        alive = self._buf["alive"][:n]
        food_pos = self._buf["food_pos"][:n]
        food_active = self._buf["food_active"][:n]
        step_arr = self._buf["step"][:n]
        agent_ids = self._buf["agent_ids"][:n]
        parent_ids = self._buf["parent_ids"][:n]
        ages = self._buf["ages"][:n]
        reward_weights = self._buf["reward_weights"][:n]
        action = self._buf["action"][:n]

        scales: dict[str, float] = {}
        # Per-replay id base for uint16 id compression. Stored in meta so the
        # loader can reconstruct absolute agent ids. 0 when quantize is off
        # or there are no valid ids in the window.
        id_base = 0
        quant_ids = False
        if self.quantize:
            pos_scale = self.world_size / 65535.0
            e_scale = self.energy_capacity / 255.0
            pos_q = np.clip(pos, 0.0, self.world_size) / pos_scale
            fp_q = np.clip(food_pos, 0.0, self.world_size) / pos_scale
            e_q = np.clip(energy, 0.0, self.energy_capacity) / e_scale
            pos_out = pos_q.astype(np.uint16)
            fp_out = fp_q.astype(np.uint16)
            energy_out = e_q.astype(np.uint8)
            # Reward weights: ±4 range covers init (N(0, ~0.5)) and long-run
            # drift with headroom. Precision ~0.03, imperceptible in the
            # histogram at ±2 display axis.
            rw_scale = 4.0 / 127.0
            rw_out = np.clip(reward_weights / rw_scale, -127, 127).astype(np.int8)
            # Actions: tanh-ish in [-1, 1]. ±1 range, precision ~0.008.
            act_scale = 1.0 / 127.0
            act_out = np.clip(action / act_scale, -127, 127).astype(np.int8)
            scales = {
                "pos": pos_scale, "food_pos": pos_scale, "energy": e_scale,
                "reward_weights": rw_scale, "action": act_scale,
            }
            # Agent/parent id compression. Absolute ids are globally monotonic
            # (next_agent_id counter); within a 1000-frame window the range
            # stays small. Encoding: 0 is the -1 sentinel ("no parent" /
            # "inactive slot"), 1..65535 stores (id - id_base + 1). Falls
            # back to int32 if the window's id range exceeds 65534.
            valids = np.concatenate([
                agent_ids[agent_ids >= 0].ravel(),
                parent_ids[parent_ids >= 0].ravel(),
            ])
            if valids.size > 0:
                id_min = int(valids.min())
                id_max = int(valids.max())
                if id_max - id_min + 1 <= 65534:
                    id_base = id_min
                    aid_out = np.where(agent_ids < 0, 0, agent_ids - id_base + 1).astype(np.uint16)
                    pid_out = np.where(parent_ids < 0, 0, parent_ids - id_base + 1).astype(np.uint16)
                    quant_ids = True
                else:
                    aid_out, pid_out = agent_ids, parent_ids
            else:
                # Empty window — emit zero-filled uint16 and id_base=0.
                id_base = 0
                aid_out = np.zeros_like(agent_ids, dtype=np.uint16)
                pid_out = np.zeros_like(parent_ids, dtype=np.uint16)
                quant_ids = True
        else:
            pos_out, fp_out, energy_out = pos, food_pos, energy
            rw_out, act_out = reward_weights, action
            aid_out, pid_out = agent_ids, parent_ids

        # Section order is the contract with the JS loader. Don't reorder
        # without updating replayLoader.ts.
        sections = [
            ("pos",         pos_out,     "uint16" if self.quantize else "float32"),
            ("angle",       angle,       "float32"),
            ("energy",      energy_out,  "uint8"  if self.quantize else "float32"),
            ("alive",       alive,       "uint8"),
            ("food_pos",    fp_out,      "uint16" if self.quantize else "float32"),
            ("food_active", food_active, "uint8"),
            ("step_nums",   step_arr,    "int32"),
            ("species",     species,     "int32"),
            ("radii",       radii,       "float32"),
            # v2 additions — identity, lineage, phenotype, action. See header.
            ("agent_ids",      aid_out,        "uint16" if quant_ids     else "int32"),
            ("parent_ids",     pid_out,        "uint16" if quant_ids     else "int32"),
            ("ages",           ages,           "int32"),
            ("reward_weights", rw_out,         "int8"   if self.quantize else "float32"),
            ("action",         act_out,        "int8"   if self.quantize else "float32"),
        ]

        bin_path = out_dir / "frames.bin"
        offsets: dict[str, dict] = {}
        with open(bin_path, "wb") as f:
            for name, arr, dtype in sections:
                arr = np.ascontiguousarray(arr).astype(dtype, copy=False)
                buf = arr.tobytes()
                offsets[name] = {
                    "offset": f.tell(),
                    "length": len(buf),
                    "dtype": dtype,
                    "shape": list(arr.shape),
                }
                f.write(buf)

        meta = {
            "version": 2,
            "start_step": start_step,
            "n_frames": int(n),
            "max_agents": int(self.max_agents),
            "food_max": int(self.food_max),
            "world_size": self.world_size,
            "quantize": self.quantize,
            "scales": scales,
            "id_base": int(id_base),
            "sections": offsets,
            "frames_bin": "frames.bin",
            "frames_bin_size": bin_path.stat().st_size,
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        size_mb = bin_path.stat().st_size / 1e6
        print(f"[replay] flushed {n} frames → {bin_path} ({size_mb:.1f} MB, start_step={start_step})")

        if self.bucket:
            self._upload_and_prune(out_dir, start_step)

        self._count = 0

    # ---------------------------- upload + prune -----------------------------

    def _upload_and_prune(self, out_dir: Path, start_step: int) -> None:
        try:
            from scripts import replay_upload
        except ImportError as e:
            print(f"[replay] upload skipped (google-cloud-storage not installed): {e}")
            return

        try:
            replay_upload.upload_replay(
                out_dir, self.exp_name, self.seed, start_step, self.bucket,
                run_tag=self.run_tag,
            )
        except Exception as e:
            print(f"[replay] upload failed: {e!r}")
            return

        try:
            # Scope retention to just this (exp, seed, run_tag) so milestones
            # from other runs in the same bucket aren't collateral damage.
            deleted = replay_upload.prune(
                self.retention_policy, self.retention_config, self.bucket,
                exp=self.exp_name, seed=self.seed, run_tag=self.run_tag,
            )
        except Exception as e:
            print(f"[replay] retention pass failed: {e!r}")
            deleted = []

        try:
            n_idx = replay_upload.rebuild_index(self.bucket)
        except Exception as e:
            print(f"[replay] index rebuild failed: {e!r}")
            return

        tag_seg = f"{self.run_tag}/" if self.run_tag else ""
        msg = (
            f"[replay] uploaded to gs://{self.bucket}/{self.exp_name}/"
            f"seed_{self.seed}/{tag_seg}step_{start_step:08d}/"
        )
        if deleted:
            msg += f" (pruned {len(deleted)} via {self.retention_policy})"
        msg += f"; index has {n_idx} replay(s)"
        print(msg)
