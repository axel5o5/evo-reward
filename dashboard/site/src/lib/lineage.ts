// Lineage index: walks the (frame × slot) grid once to build
// agent_id → parent_id and agent_id → [child_ids]. Used by AgentInspector
// to surface "is this agent's lineage actually winning?" — direct offspring
// count + generation depth from founder.
//
// Scope is the replay window: ancestors born before recording started are
// invisible (we only know up to the agent whose parent_id falls outside the
// id space we observed). lineageDepth() returns the number of ancestors we
// can chain to *within the window*, plus a flag for whether we hit a real
// founder (parent_id < 0) or just ran out of visibility.

import { ReplayData } from "./replayLoader";

export interface LineageIndex {
  // agent_id → parent_id. parent_id < 0 means "founder" (no parent).
  // Missing keys mean the agent was never observed alive in this replay.
  idToParent: Map<number, number>;
  // agent_id → set of child agent_ids. Sets dedupe — without this we'd
  // double-count any child that survived multiple frames (which is most
  // children).
  idToChildren: Map<number, Set<number>>;
}

export function buildLineageIndex(data: ReplayData): LineageIndex | null {
  if (!data.agentIds || !data.parentIds) return null;
  const n = data.meta.n_frames;
  const N = data.meta.max_agents;
  const idToParent = new Map<number, number>();
  const idToChildren = new Map<number, Set<number>>();
  for (let f = 0; f < n; f++) {
    const base = f * N;
    for (let s = 0; s < N; s++) {
      if (data.alive[base + s] === 0) continue;
      const id = data.agentIds[base + s];
      if (id < 0) continue;
      const pid = data.parentIds[base + s];
      // First-seen wins: an agent's parent is fixed at birth, so any later
      // frame would record the same value. Skipping if already mapped saves
      // a tiny constant per cell.
      if (!idToParent.has(id)) {
        idToParent.set(id, pid);
        if (pid >= 0) {
          let kids = idToChildren.get(pid);
          if (!kids) {
            kids = new Set();
            idToChildren.set(pid, kids);
          }
          kids.add(id);
        }
      }
    }
  }
  return { idToParent, idToChildren };
}

export interface LineageDepth {
  // Number of ancestor hops we could chain *within the replay window*.
  // 0 means we couldn't even resolve the agent's parent (parent born
  // before recording started, or agent itself is a founder).
  depthInWindow: number;
  // True iff the chain terminated at a real founder (parent_id < 0), not
  // because we ran out of in-window ancestors. Lets the UI render
  // "5 generations from founder" vs "≥5 generations (chain runs off the
  // start of the window)".
  reachedFounder: boolean;
}

export function lineageDepth(
  index: LineageIndex,
  agentId: number,
): LineageDepth {
  let depth = 0;
  let reachedFounder = false;
  let cur = agentId;
  // Defensive: cap at 1024 hops to ensure no infinite loop if a cycle
  // somehow makes it into the data. Real lineage chains are far shorter.
  for (let i = 0; i < 1024; i++) {
    const parent = index.idToParent.get(cur);
    if (parent === undefined) break; // chain runs off the window
    if (parent < 0) {
      reachedFounder = true;
      break;
    }
    depth++;
    cur = parent;
  }
  return { depthInWindow: depth, reachedFounder };
}

export function offspringCount(
  index: LineageIndex,
  agentId: number,
): number {
  return index.idToChildren.get(agentId)?.size ?? 0;
}
