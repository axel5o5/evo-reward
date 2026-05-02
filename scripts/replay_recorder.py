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

Binary format is versioned via meta.json.version.
  v1 — geometry only.
  v2 — adds per-frame identity/lineage/phenotype tracks (agent_ids, parent_ids,
       ages, reward_weights, action).
  v3 — adds per-genome reward storage for MLP/temporal runs and tags the
       genome architecture in meta. Linear runs are unchanged (still write
       per-frame `reward_weights`); MLP/temporal runs drop that 4-vector
       section (it's frozen at init for non-linear genomes and uninformative)
       and write `reward_genomes_byid` + `reward_genomes_idmap` instead —
       one flat genome row per unique agent_id seen during the window.
       meta gains `genome_arch` ("linear" | "mlp" | "temporal"),
       `genome_shape` (architecture-specific dims), and `genome_layout`
       (offset/shape table the loader uses to re-nest the flat row into
       per-layer kernels/biases). Storage at defaults: ~240 KB extra for
       hidden_size=8 MLP, ~1.9 MB for k=10/h=16 temporal.

Dtypes below assume quantize=True:
  agent_ids      uint16  (n_frames, max_agents)     stored = (id - id_base + 1),
                                                    0 = sentinel (-1). id_base
                                                    lives in meta.id_base.
                                                    Falls back to int32 when
                                                    the window's id range
                                                    exceeds 65534.
  parent_ids     uint16  (n_frames, max_agents)     same encoding as agent_ids
  ages           int32   (n_frames, max_agents)     birth_step = step - age
  reward_weights int8    (n_frames, max_agents, 4)  scale = 4/127 (covers ±4)
                                                    — linear genome only (v3)
  action         int8    (n_frames, max_agents, 2)  scale = 1/127 (covers ±1)
  reward_genomes_byid    float32 (n_unique, genome_dim)  — v3, MLP/temporal
  reward_genomes_idmap   int32   (n_unique,)             — v3, MLP/temporal
Per-frame (not static) because slot reuse after death replaces the occupant
mid-window — a static section would misattribute to the previous tenant.
Unquantized replays keep everything as float32 / int32.

v2 storage at defaults (length=1000, max_agents=500): ~14 MB (vs ~8 MB v1).
v3 linear: same size as v2. v3 MLP (h=8): ~14 MB + ~240 KB.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from src.save_schedule import parse_schedule, interval_at


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
        raw_interval = config.get("replay_record_interval_steps", 0) or 0
        # Treat 0 as "disabled"; otherwise build a schedule (int or list).
        if isinstance(raw_interval, int) and raw_interval == 0:
            self.schedule = [(0, 0)]
            self.enabled = False
        else:
            self.schedule = parse_schedule(raw_interval, default_interval=0)
            self.enabled = self.schedule[0][1] > 0
        # Legacy attr kept for any external code reading recorder.interval.
        self.interval = self.schedule[0][1]
        self.length = int(config.get("replay_record_length_steps", 1000) or 0)
        self.enabled = self.enabled and self.length > 0
        if not self.enabled:
            return
        # Stateful next-flush boundary. Updated after each flush.
        self.next_boundary = self.schedule[0][1]

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

        # v3 genome capture. For linear runs we keep the v2 per-frame
        # reward_weights pipeline. For MLP/temporal runs we ditch that
        # (the 4-vector field is frozen at init for those genomes and
        # uninformative) and snapshot the flat genome the first frame
        # we see each agent_id during a recording window. seen_genomes
        # is reset on each flush.
        self.genome_arch = str(config.get("reward_type", "linear"))
        self.genome_dim = 0
        self.genome_shape: dict = {}
        self.genome_layout: list[dict] = []
        if self.genome_arch in ("mlp", "temporal"):
            self._init_genome_layout(config)
        self.seen_genomes: dict[int, np.ndarray] = {}

        # local_out_root is already `<out_dir>/<exp>/seed_<N>[/<run_tag>]/replays`
        # when called from the runner, so don't re-append exp/seed here.
        self.local_root = Path(local_out_root)
        self.local_root.mkdir(parents=True, exist_ok=True)

        self._alloc_buffer()

    # ---------------------------- genome layout (v3) -------------------------

    def _init_genome_layout(self, config: dict) -> None:
        """Compute the flat-row layout for an MLP/temporal genome.

        Builds it from a freshly initialized template so the exact leaf
        ordering matches `jax.tree_util.tree_leaves` — which is also what
        `ravel_pytree` uses when the runner flattens per-slot params at
        capture time. Strips the leading 'params' key so the layout maps
        cleanly to the dashboard's `MlpGenome` interface (which doesn't
        carry that wrapper).
        """
        import jax
        import jax.tree_util as jtu

        if self.genome_arch == "mlp":
            from src.reward import init_mlp_genome
            template = init_mlp_genome(jax.random.PRNGKey(0), config)
            hidden = int(config["mlp_hidden_size"])
            self.genome_shape = {"input_dim": 4, "hidden_size": hidden, "output_dim": 1}
        else:
            from src.reward import init_temporal_genome
            template = init_temporal_genome(jax.random.PRNGKey(0), config)
            k = int(config["reward_context_window"])
            hidden = int(config["temporal_hidden_size"])
            self.genome_shape = {
                "input_dim": k * 4,
                "context_window": k,
                "hidden_size": hidden,
                "output_dim": 1,
            }

        leaves_with_paths, _ = jtu.tree_flatten_with_path(template)
        offset = 0
        for path, leaf in leaves_with_paths:
            keys = []
            for p in path:
                key = getattr(p, "key", None)
                if key is None:
                    key = str(p)
                keys.append(str(key))
            # Strip the Flax 'params' wrapper so dashboard consumers see a
            # bare {Dense_0: {...}, Dense_1: {...}, ...} structure.
            if keys and keys[0] == "params":
                keys = keys[1:]
            shape = list(leaf.shape)
            size = int(np.prod(shape)) if shape else 1
            self.genome_layout.append({"path": keys, "shape": shape, "offset": offset})
            offset += size
        self.genome_dim = offset

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

        # Resume catch-up: if start_step jumped past one or more boundaries
        # (e.g., resumed from a checkpoint mid-run), fast-forward.
        while step_after > self.next_boundary:
            self.next_boundary += interval_at(self.schedule, self.next_boundary)

        # Upcoming boundary we're racing toward. Stored on self so a
        # tapered schedule (interval changes mid-run) Just Works.
        upcoming = self.next_boundary
        in_window = (upcoming - step_after) < self.length

        if in_window:
            self._capture(sim_state, step_after)
        if step_after == upcoming and self._count > 0:
            self._flush(sim_state)
            self.next_boundary = step_after + interval_at(self.schedule, step_after)

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

        # v3: snapshot per-slot MLP/temporal genome the first frame we see
        # each agent_id during this window. Slot reuse is handled correctly
        # because the dict is keyed by id, not slot. Genomes don't change
        # over an agent's lifetime (mutated only at birth), so one row is
        # exact, not a sample.
        if self.genome_arch in ("mlp", "temporal"):
            self._capture_genomes(sim_state, i)

        self._count += 1

    def _capture_genomes(self, sim_state, frame_idx: int) -> None:
        """Snapshot any newly-seen agent ids' flat genomes."""
        import jax.tree_util as jtu

        ids = self._buf["agent_ids"][frame_idx]
        active = self._buf["alive"][frame_idx]
        new_slots = []
        new_ids = []
        for slot in range(self.max_agents):
            if not active[slot]:
                continue
            aid = int(ids[slot])
            if aid < 0 or aid in self.seen_genomes:
                continue
            new_slots.append(slot)
            new_ids.append(aid)
        if not new_slots:
            return

        # Pull the whole stacked PyTree once and concat per-slot rows along
        # the genome axis. Using tree_leaves matches ravel_pytree's leaf
        # ordering (genome_layout was built from the same traversal at
        # __init__), so the flat row maps onto layout offsets exactly.
        params = (
            sim_state.reward_mlp_params
            if self.genome_arch == "mlp"
            else sim_state.reward_temporal_params
        )
        leaves = jtu.tree_leaves(params)
        flat_per_slot = np.concatenate(
            [np.asarray(leaf).reshape(self.max_agents, -1) for leaf in leaves],
            axis=1,
        )
        for slot, aid in zip(new_slots, new_ids):
            self.seen_genomes[aid] = flat_per_slot[slot].astype(np.float32, copy=True)

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
            # v2 additions — identity, lineage, action. The reward_weights
            # section only goes in for linear runs (v3 drops it for
            # MLP/temporal — the 4-vector is uninformative there).
            ("agent_ids",      aid_out,        "uint16" if quant_ids     else "int32"),
            ("parent_ids",     pid_out,        "uint16" if quant_ids     else "int32"),
            ("ages",           ages,           "int32"),
        ]
        if self.genome_arch == "linear":
            sections.append(
                ("reward_weights", rw_out, "int8" if self.quantize else "float32")
            )
        sections.append(
            ("action", act_out, "int8" if self.quantize else "float32"),
        )

        # v3 genome-by-id sections. One row per unique agent_id seen during
        # the window, keyed by reward_genomes_idmap. Float32 for now — the
        # storage is small (240 KB for h=8 MLP at defaults), so we don't
        # need quantization yet.
        if self.genome_arch in ("mlp", "temporal") and self.seen_genomes:
            ids_sorted = sorted(self.seen_genomes.keys())
            genomes_byid = np.stack([self.seen_genomes[i] for i in ids_sorted]).astype(np.float32)
            genomes_idmap = np.array(ids_sorted, dtype=np.int32)
            sections.append(("reward_genomes_byid", genomes_byid, "float32"))
            sections.append(("reward_genomes_idmap", genomes_idmap, "int32"))

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
            "version": 3,
            "start_step": start_step,
            "n_frames": int(n),
            "max_agents": int(self.max_agents),
            "food_max": int(self.food_max),
            "world_size": self.world_size,
            "quantize": self.quantize,
            "scales": scales,
            "id_base": int(id_base),
            "genome_arch": self.genome_arch,
            "genome_shape": self.genome_shape,
            "genome_layout": self.genome_layout,
            "genome_dim": int(self.genome_dim),
            "sections": offsets,
            "frames_bin": "frames.bin",
            "frames_bin_size": bin_path.stat().st_size,
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        size_mb = bin_path.stat().st_size / 1e6
        n_genomes = len(self.seen_genomes) if self.genome_arch != "linear" else 0
        suffix = f", {n_genomes} genome(s)" if n_genomes else ""
        print(f"[replay] flushed {n} frames → {bin_path} ({size_mb:.1f} MB, start_step={start_step}{suffix})")

        if self.bucket:
            self._upload_and_prune(out_dir, start_step)

        self._count = 0
        self.seen_genomes = {}

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
