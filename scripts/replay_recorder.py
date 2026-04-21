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
    ):
        self.interval = int(config.get("replay_record_interval_steps", 0) or 0)
        self.length = int(config.get("replay_record_length_steps", 1000) or 0)
        self.enabled = self.interval > 0 and self.length > 0
        if not self.enabled:
            return

        self.exp_name = exp_name
        self.seed = seed
        self.max_agents = int(config["prey_cap"]) + int(config["predator_cap"])
        self.food_max = int(config["food_max"])
        self.world_size = float(config["world_size"])
        self.energy_capacity = float(config.get("energy_capacity", 1000.0))
        self.quantize = bool(config.get("replay_quantize", True))
        self.bucket = bucket
        self.retention_policy = str(config.get("replay_retention_policy", "last_n"))
        self.retention_config = dict(config.get("replay_retention_config", {"keep_last_n": 10}))

        self.local_root = Path(local_out_root) / exp_name / f"seed_{seed}"
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

        scales: dict[str, float] = {}
        if self.quantize:
            pos_scale = self.world_size / 65535.0
            e_scale = self.energy_capacity / 255.0
            pos_q = np.clip(pos, 0.0, self.world_size) / pos_scale
            fp_q = np.clip(food_pos, 0.0, self.world_size) / pos_scale
            e_q = np.clip(energy, 0.0, self.energy_capacity) / e_scale
            pos_out = pos_q.astype(np.uint16)
            fp_out = fp_q.astype(np.uint16)
            energy_out = e_q.astype(np.uint8)
            scales = {"pos": pos_scale, "food_pos": pos_scale, "energy": e_scale}
        else:
            pos_out, fp_out, energy_out = pos, food_pos, energy

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
            "version": 1,
            "start_step": start_step,
            "n_frames": int(n),
            "max_agents": int(self.max_agents),
            "food_max": int(self.food_max),
            "world_size": self.world_size,
            "quantize": self.quantize,
            "scales": scales,
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
                out_dir, self.exp_name, self.seed, start_step, self.bucket
            )
        except Exception as e:
            print(f"[replay] upload failed: {e!r}")
            return

        try:
            deleted = replay_upload.prune(
                self.retention_policy, self.retention_config, self.bucket,
            )
        except Exception as e:
            print(f"[replay] retention pass failed: {e!r}")
            deleted = []

        try:
            n_idx = replay_upload.rebuild_index(self.bucket)
        except Exception as e:
            print(f"[replay] index rebuild failed: {e!r}")
            return

        msg = f"[replay] uploaded to gs://{self.bucket}/{self.exp_name}/seed_{self.seed}/step_{start_step:08d}/"
        if deleted:
            msg += f" (pruned {len(deleted)} via {self.retention_policy})"
        msg += f"; index has {n_idx} replay(s)"
        print(msg)
