import { useEffect, useState } from "react";

// Schema mirrors scripts/gcp_monitor.py -> run() output.
type VM = {
  name: string;
  status: string;                // RUNNING | TERMINATED | STOPPED | MISSING | ...
  zone: string | null;
  machine_type: string | null;
  provisioning: string;          // SPOT | STANDARD | UNKNOWN
  created_at: string | null;
  last_started_at: string | null;
  runtime_hours_current: number | null;
  hourly_rate_usd: number | null;
  estimated_current_run_usd: number | null;
  labels: Record<string, string>;     // { experiment, phase, seed }
};

type Checkpoints = {
  bucket: string;
  count: number;
  latest_step: number | null;
  latest_age_hours: number | null;
  total_gb: number | null;
};

type Costs = {
  compute_current_run: number;
  nat_since_active: number | null;
  storage_current: number;
  live_estimate_total: number;
  billing_actual_usd: number | null;
  billing_as_of: string | null;
  month_to_date_usd: number | null;
};

type WeightPair = [number, number];  // [mean, std]

type Training = {
  experiment_name: string;
  seed: number;
  step: number;
  total_steps: number;
  progress_frac: number;
  sps: number;
  eta_hours: number | null;
  population: {
    prey?: number; pred?: number; food?: number; mean_energy?: number;
  };
  reward_weights: {
    prey?: Record<string, WeightPair>;
    pred?: Record<string, WeightPair>;
  };
  progress_file_age_hours: number;
  evolution_detected: boolean;
};

type WorkerError = { stage: string; message: string };

type Payload = {
  updated_at: string;
  project_id: string;
  vm: VM;
  checkpoints: Checkpoints;
  costs: Costs;
  training: Training | null;
  errors: WorkerError[];
};

// Static link to the config file on GitHub's main branch. The VM's
// experiment label matches configs/<experiment>.yaml by convention.
const GH_REPO_URL = "https://github.com/axel5o5/evo-reward";
function configUrlFor(experiment: string): string {
  return `${GH_REPO_URL}/blob/main/configs/${experiment}.yaml`;
}

// Default to the Vercel proxy route (api/status.ts), which reads from
// the private gcp-status branch with a server-side PAT. Local dev can
// override with VITE_GCP_STATUS_URL=/gcp-status.json + a file in public/.
const STATUS_URL: string = import.meta.env.VITE_GCP_STATUS_URL || "/api/status";
const POLL_MS = 30_000;
const STALE_MINUTES = 20;

const FIXTURE: Payload = {
  updated_at: new Date().toISOString(),
  project_id: "evo-reward",
  vm: {
    name: "evo-reward-gpu", status: "UNKNOWN", zone: null, machine_type: null,
    provisioning: "UNKNOWN", created_at: null, last_started_at: null,
    runtime_hours_current: null, hourly_rate_usd: null, estimated_current_run_usd: null,
    labels: {},
  },
  checkpoints: {
    bucket: "evo-reward-ckpts", count: 0,
    latest_step: null, latest_age_hours: null, total_gb: null,
  },
  costs: {
    compute_current_run: 0, nat_since_active: null, storage_current: 0,
    live_estimate_total: 0, billing_actual_usd: null, billing_as_of: null,
    month_to_date_usd: null,
  },
  training: null,
  errors: [{ stage: "config",
    message: "VITE_GCP_STATUS_URL not set — see dashboard/ops/GCP_MONITOR.md." }],
};

function usd(n: number | null | undefined): string {
  if (n == null) return "—";
  return `$${n.toFixed(2)}`;
}

function minutesAgo(iso: string): number {
  return (Date.now() - new Date(iso).getTime()) / 60_000;
}

function hoursToDuration(h: number | null): string {
  if (h == null) return "—";
  if (h < 1) return `${Math.round(h * 60)}m`;
  if (h < 24) return `${h.toFixed(1)}h`;
  return `${Math.floor(h / 24)}d ${Math.floor(h % 24)}h`;
}

function statusClasses(status: string): string {
  if (status === "RUNNING") return "bg-green-100 dark:bg-green-900/40 text-green-700 dark:text-green-300";
  if (status === "TERMINATED" || status === "STOPPED") return "bg-amber-100 dark:bg-amber-900/40 text-amber-700 dark:text-amber-300";
  if (status === "MISSING") return "bg-gray-100 dark:bg-gray-800 text-gray-500 dark:text-gray-400";
  return "bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300";
}

function Card({ title, children, footer }: {
  title: React.ReactNode;
  children: React.ReactNode;
  footer?: React.ReactNode;
}) {
  return (
    <div className="rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 overflow-hidden">
      <div className="px-4 py-2 border-b border-gray-100 dark:border-gray-800 text-xs font-semibold uppercase tracking-wide text-gray-500 dark:text-gray-400">
        {title}
      </div>
      <div className="p-4">{children}</div>
      {footer && (
        <div className="px-4 py-2 border-t border-gray-100 dark:border-gray-800 text-xs text-gray-500 dark:text-gray-400">
          {footer}
        </div>
      )}
    </div>
  );
}

function Row({ label, value, title, mono = true }: {
  label: React.ReactNode;
  value: React.ReactNode;
  title?: string;
  mono?: boolean;
}) {
  return (
    <div className="flex justify-between py-1 text-sm" title={title}>
      <span className="text-gray-600 dark:text-gray-400">{label}</span>
      <span className={`text-gray-900 dark:text-gray-100 ${mono ? "font-mono" : ""}`}>{value}</span>
    </div>
  );
}

function VMCard({ vm }: { vm: VM }) {
  const provLabel = vm.provisioning === "SPOT" ? "spot" :
                    vm.provisioning === "STANDARD" ? "on-demand" :
                    vm.provisioning.toLowerCase();
  const experiment = vm.labels.experiment;
  const phase = vm.labels.phase;
  const seed = vm.labels.seed;
  const hasRunIdentity = !!(experiment || phase || seed);

  return (
    <Card title="VM">
      <div className="flex items-center gap-2 mb-3">
        <span className={`px-2 py-0.5 text-xs rounded-full ${statusClasses(vm.status)}`}>
          {vm.status}
        </span>
        <span className="font-mono text-sm text-gray-800 dark:text-gray-200">{vm.name}</span>
        {vm.provisioning !== "UNKNOWN" && (
          <span className="text-xs text-gray-500 dark:text-gray-400">({provLabel})</span>
        )}
      </div>

      {hasRunIdentity && (
        <div className="mb-3 pb-3 border-b border-gray-100 dark:border-gray-800 text-xs">
          <div className="text-gray-500 dark:text-gray-400 mb-1">Run</div>
          <div className="font-mono text-gray-900 dark:text-gray-100">
            {experiment || "?"}
            {phase && <span className="text-gray-400"> · phase {phase}</span>}
            {seed !== undefined && <span className="text-gray-400"> · seed {seed}</span>}
          </div>
          {experiment && (
            <a
              href={configUrlFor(experiment)}
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-600 dark:text-blue-400 hover:underline text-[11px]"
            >
              configs/{experiment}.yaml ↗
            </a>
          )}
        </div>
      )}

      <Row label="Zone" value={vm.zone || "—"} />
      <Row label="Machine" value={vm.machine_type || "—"} />
      <Row label="Runtime (current)" value={hoursToDuration(vm.runtime_hours_current)} />
      <Row label="Hourly rate" value={vm.hourly_rate_usd != null ? `$${vm.hourly_rate_usd.toFixed(3)}/h` : "—"} />
      <Row label="Current run cost" value={usd(vm.estimated_current_run_usd)} />
    </Card>
  );
}

function WeightLine({
  label, pair, wantSign, note,
}: {
  label: string;
  pair: WeightPair | undefined;
  wantSign?: "+" | "-" | null;
  note: string;
}) {
  if (!pair) return null;
  const [m, s] = pair;
  let tone = "text-gray-500 dark:text-gray-500";
  if (wantSign === "+") tone = m > 0 ? "text-green-600 dark:text-green-400" : "text-amber-600 dark:text-amber-400";
  else if (wantSign === "-") tone = m < 0 ? "text-green-600 dark:text-green-400" : "text-amber-600 dark:text-amber-400";
  return (
    <div className="flex justify-between text-xs font-mono py-0.5">
      <span className="text-gray-600 dark:text-gray-400 w-14">{label}</span>
      <span className="w-20 text-right text-gray-900 dark:text-gray-100">
        {m >= 0 ? "+" : ""}{m.toFixed(3)} ± {s.toFixed(3)}
      </span>
      <span className={`flex-1 pl-3 text-[10px] ${tone}`}>{note}</span>
    </div>
  );
}

function TrainingCard({ t }: { t: Training }) {
  const pct = (t.progress_frac * 100);
  const pop = t.population;
  const prey = t.reward_weights.prey;
  const pred = t.reward_weights.pred;
  const fresh = t.progress_file_age_hours < 0.2;

  return (
    <Card
      title={
        <span>
          Training
          {t.evolution_detected && (
            <span className="ml-2 text-[10px] px-1.5 py-0.5 rounded bg-green-100 text-green-700 dark:bg-green-900/40 dark:text-green-300 normal-case">
              evolution detected
            </span>
          )}
        </span>
      }
      footer={`progress.json age: ${hoursToDuration(t.progress_file_age_hours)}${fresh ? "" : " — may be stale"}`}
    >
      <div className="mb-3">
        <div className="flex justify-between text-xs text-gray-600 dark:text-gray-400 mb-1">
          <span>Step {t.step.toLocaleString()} / {t.total_steps.toLocaleString()}</span>
          <span className="font-mono">{pct.toFixed(2)}%</span>
        </div>
        <div className="h-2 rounded bg-gray-200 dark:bg-gray-800 overflow-hidden">
          <div
            className="h-full bg-blue-500 dark:bg-blue-600 transition-all"
            style={{ width: `${Math.min(100, Math.max(0, pct))}%` }}
          />
        </div>
      </div>

      <Row
        label="Rate"
        value={<span>{t.sps.toFixed(1)} sps
          {t.eta_hours != null && (
            <span className="text-gray-400 ml-1">· ETA {hoursToDuration(t.eta_hours)}</span>
          )}
        </span>}
      />
      {pop && (
        <>
          <Row
            label="Population"
            value={<span>prey {pop.prey ?? "—"} · pred {pop.pred ?? "—"} · food {pop.food ?? "—"} · E {pop.mean_energy?.toFixed(1) ?? "—"}</span>}
            mono={false}
          />
          {(pop.prey === 450 || pop.pred === 50) && (
            <div className="text-[10px] text-amber-600 dark:text-amber-400 pl-1 -mt-1 pb-1" title="K&D Table 1 steady-state: ~349 prey, ~23 predators. Populations pinned at caps indicate oscillations haven't developed yet — expected before ~3M steps, concerning after.">
              at cap (K&D steady-state: ~349 prey / ~23 pred)
            </div>
          )}
        </>
      )}

      {prey && (
        <div className="mt-3 pt-2 border-t border-gray-100 dark:border-gray-800">
          <div className="text-[10px] uppercase tracking-wide text-gray-500 dark:text-gray-400 mb-1">Prey reward weights</div>
          <WeightLine label="w_eat"  pair={prey.eat}  wantSign="+" note="food reward — universal >0" />
          <WeightLine label="w_act"  pair={prey.act}  wantSign="+" note="consistently >0 (K&D §4.2)" />
          <WeightLine label="w_prey" pair={prey.prey} wantSign="+" note="social affiliation — >0 in 3/5 seeds" />
          <WeightLine label="w_pred" pair={prey.pred} wantSign={null} note="fear — bimodal: <0 in 3/5 (fear), >0 in 2/5 (sociality compensates)" />
        </div>
      )}

      {pred && (
        <div className="mt-3 pt-2 border-t border-gray-100 dark:border-gray-800">
          <div className="text-[10px] uppercase tracking-wide text-gray-500 dark:text-gray-400 mb-1">Predator reward weights</div>
          <WeightLine label="w_eat"  pair={pred.eat}  wantSign="+" note="prey-eating reward — >0 medium mouth" />
          <WeightLine label="w_act"  pair={pred.act}  wantSign={null} note="bimodal ± by seed" />
          <WeightLine label="w_prey" pair={pred.prey} wantSign="+" note="prey attraction — >0 most seeds, near 0 if prey.w_pred>0" />
          <WeightLine label="w_pred" pair={pred.pred} wantSign="+" note="social (follow conspecifics to prey-rich regions)" />
        </div>
      )}
    </Card>
  );
}

function CheckpointsCard({ c }: { c: Checkpoints }) {
  const fresh = c.latest_age_hours != null && c.latest_age_hours < 0.5;
  return (
    <Card title="Checkpoints" footer={`gs://${c.bucket}/results/`}>
      <Row label="Count" value={c.count.toString()} />
      <Row
        label="Latest step"
        value={c.latest_step != null ? c.latest_step.toLocaleString() : "—"}
      />
      <Row
        label="Last saved"
        value={
          <span className={fresh ? "text-green-600 dark:text-green-400" : ""}>
            {c.latest_age_hours != null ? `${hoursToDuration(c.latest_age_hours)} ago` : "—"}
          </span>
        }
      />
      <Row label="Bucket size" value={c.total_gb != null ? `${c.total_gb.toFixed(2)} GB` : "—"} />
    </Card>
  );
}

function CostsCard({ costs }: { costs: Costs }) {
  return (
    <Card title="Costs (USD)">
      <div className="mb-3 pb-3 border-b border-gray-100 dark:border-gray-800">
        <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">Live estimate (now)</div>
        <div className="text-2xl font-mono font-semibold text-gray-900 dark:text-gray-100">
          {usd(costs.live_estimate_total)}
        </div>
      </div>
      <Row
        label="VM compute (current run)"
        value={usd(costs.compute_current_run)}
        title="VM runtime since last_started × hourly rate. Resets at preemption."
      />
      <Row
        label="Cloud NAT (24/7)"
        value={usd(costs.nat_since_active)}
        title="NAT gateway charges ~$0.044/hr whether the VM is up or not. Set nat_active_since in the config."
      />
      <Row
        label="GCS storage (MTD)"
        value={usd(costs.storage_current)}
        title="Bucket size × $/GB-month × fraction of month elapsed."
      />
      <div className="mt-3 pt-3 border-t border-gray-100 dark:border-gray-800">
        <Row
          label={<span>Billing actual <span className="text-[10px] text-gray-400">(~24h lag)</span></span>}
          value={usd(costs.billing_actual_usd)}
          title="From BigQuery billing export — lags ~24h. null if not configured."
        />
        <Row label="Month to date" value={usd(costs.month_to_date_usd)} />
      </div>
    </Card>
  );
}

// ───────────────────────────────────────────────────────────────────────
// Control panel — PIN-gated buttons that hit /api/action, which forwards
// to the gcp-action GitHub workflow. No GCP creds in the browser or on
// Vercel; the workflow has them via WIF.

type Action = "stop" | "start" | "restart" | "delete";

type ActionState =
  | { kind: "idle" }
  | { kind: "dispatching"; action: Action }
  | { kind: "success"; action: Action; requestId: string; at: number }
  | { kind: "error"; action: Action; message: string; at: number };

// Reasonable rules for when each action makes sense given current VM state.
function enabledFor(status: string): Record<Action, boolean> {
  const running = status === "RUNNING";
  const stopped = status === "TERMINATED" || status === "STOPPED";
  const exists = status !== "MISSING" && status !== "UNKNOWN";
  return {
    stop: running,
    start: stopped,
    restart: running,
    delete: exists,
  };
}

const ACTION_META: Record<Action, {
  label: string;
  hint: string;
  destructive: boolean;
  confirmText?: string;
}> = {
  stop: {
    label: "Stop",
    hint: "Pauses the VM. Disk + checkpoints survive. Billing drops to ~$0.02/hr.",
    destructive: false,
  },
  start: {
    label: "Start",
    hint: "Wakes a stopped VM in its existing zone. Training auto-resumes from last checkpoint.",
    destructive: false,
  },
  restart: {
    label: "Restart",
    hint: "Stop + Start. Clears stuck CUDA / tmux / memory without re-provisioning.",
    destructive: false,
  },
  delete: {
    label: "Delete",
    hint: "Destroys the VM. If the local orchestrator is running, it'll respawn in a fresh zone.",
    destructive: true,
    confirmText: "Type DELETE to confirm",
  },
};

const PIN_STORAGE_KEY = "evo-dashboard-pin";

function ControlPanel({ vmStatus }: { vmStatus: string }) {
  const [pin, setPin] = useState<string>(() => {
    if (typeof window === "undefined") return "";
    return sessionStorage.getItem(PIN_STORAGE_KEY) || "";
  });
  const [state, setState] = useState<ActionState>({ kind: "idle" });
  const [pendingConfirm, setPendingConfirm] = useState<Action | null>(null);
  const [confirmInput, setConfirmInput] = useState("");

  const enabled = enabledFor(vmStatus);

  async function dispatch(action: Action) {
    if (!pin) {
      setState({ kind: "error", action, message: "PIN required", at: Date.now() });
      return;
    }
    sessionStorage.setItem(PIN_STORAGE_KEY, pin);
    setState({ kind: "dispatching", action });
    try {
      const res = await fetch("/api/action", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pin, action }),
      });
      const data = await res.json().catch(() => ({} as { request_id?: string; error?: string }));
      if (!res.ok) {
        setState({
          kind: "error", action,
          message: data.error || `HTTP ${res.status}`,
          at: Date.now(),
        });
        return;
      }
      setState({
        kind: "success", action,
        requestId: data.request_id || "?",
        at: Date.now(),
      });
    } catch (e) {
      setState({ kind: "error", action, message: String(e), at: Date.now() });
    }
  }

  function handleClick(action: Action) {
    if (ACTION_META[action].destructive) {
      setPendingConfirm(action);
      setConfirmInput("");
    } else {
      dispatch(action);
    }
  }

  function confirmAndDispatch() {
    if (!pendingConfirm) return;
    const meta = ACTION_META[pendingConfirm];
    const expected = meta.label.toUpperCase();
    if (confirmInput !== expected) return;
    const action = pendingConfirm;
    setPendingConfirm(null);
    setConfirmInput("");
    dispatch(action);
  }

  return (
    <Card title="Control panel" footer="Actions run via GitHub Actions using WIF — audit trail in the repo's Actions tab.">
      <div className="mb-3">
        <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">PIN</label>
        <input
          type="password"
          value={pin}
          onChange={(e) => setPin(e.target.value)}
          placeholder="shared PIN"
          className="w-full border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 rounded px-2 py-1 text-sm font-mono focus:outline-none focus:border-blue-400 dark:text-gray-100"
          autoComplete="off"
        />
      </div>

      <div className="grid grid-cols-2 gap-2 mb-3">
        {(Object.keys(ACTION_META) as Action[]).map((a) => {
          const meta = ACTION_META[a];
          const disabled = !enabled[a] || state.kind === "dispatching" || !pin;
          const busy = state.kind === "dispatching" && state.action === a;
          const base = "px-3 py-2 text-sm rounded transition disabled:opacity-40 disabled:cursor-not-allowed";
          const cls = meta.destructive
            ? `${base} bg-red-600 hover:bg-red-700 text-white disabled:bg-red-600`
            : `${base} bg-blue-600 hover:bg-blue-700 text-white disabled:bg-blue-600`;
          return (
            <button
              key={a}
              onClick={() => handleClick(a)}
              disabled={disabled}
              className={cls}
              title={meta.hint}
            >
              {busy ? "…" : meta.label}
            </button>
          );
        })}
      </div>

      {pendingConfirm && (
        <div className="mt-2 p-2 rounded border border-red-300 dark:border-red-800 bg-red-50 dark:bg-red-950/30 text-xs">
          <div className="mb-1 text-red-700 dark:text-red-300">
            {ACTION_META[pendingConfirm].confirmText}
          </div>
          <div className="flex gap-2">
            <input
              type="text"
              value={confirmInput}
              onChange={(e) => setConfirmInput(e.target.value)}
              placeholder={ACTION_META[pendingConfirm].label.toUpperCase()}
              className="flex-1 border border-red-300 dark:border-red-800 bg-white dark:bg-gray-900 rounded px-2 py-1 font-mono focus:outline-none focus:border-red-500 dark:text-gray-100"
              autoFocus
            />
            <button
              onClick={confirmAndDispatch}
              disabled={confirmInput !== ACTION_META[pendingConfirm].label.toUpperCase()}
              className="px-3 py-1 bg-red-600 text-white rounded disabled:opacity-40 disabled:cursor-not-allowed"
            >
              Confirm
            </button>
            <button
              onClick={() => { setPendingConfirm(null); setConfirmInput(""); }}
              className="px-3 py-1 bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-200 rounded"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {state.kind === "success" && (
        <div className="text-xs text-green-600 dark:text-green-400">
          Dispatched <span className="font-mono">{state.action}</span>
          <span className="text-gray-400"> · req {state.requestId}</span>
          <span className="text-gray-400"> · reflects in ~1 min</span>
        </div>
      )}
      {state.kind === "error" && (
        <div className="text-xs text-red-600 dark:text-red-400">
          {state.action}: {state.message}
        </div>
      )}
      {state.kind === "idle" && (
        <div className="text-[10px] text-gray-400 dark:text-gray-500">
          PIN is cached in this tab's sessionStorage. Closes when you close the tab.
        </div>
      )}
    </Card>
  );
}

export default function GcpMonitor() {
  const [payload, setPayload] = useState<Payload | null>(null);
  const [fetchError, setFetchError] = useState<string | null>(null);
  const [lastFetchedAt, setLastFetchedAt] = useState<number>(0);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const res = await fetch(`${STATUS_URL}?t=${Date.now()}`, { cache: "no-store" });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = (await res.json()) as Payload;
        if (!cancelled) { setPayload(data); setFetchError(null); setLastFetchedAt(Date.now()); }
      } catch (e) {
        if (!cancelled) {
          setFetchError(String(e));
          // Render the fixture only if we've never had real data — use
          // the updater form so we don't clobber a previously-good fetch
          // after a transient network blip.
          setPayload((prev) => prev ?? FIXTURE);
        }
      }
    }
    load();
    const id = setInterval(load, POLL_MS);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  const isStale = payload ? minutesAgo(payload.updated_at) > STALE_MINUTES : false;

  return (
    <div className="max-w-6xl mx-auto py-10 px-6">
      <div className="flex items-baseline justify-between mb-2">
        <h1 className="text-3xl font-bold">GCP Monitor</h1>
        {payload && (
          <div className="text-xs text-gray-500 dark:text-gray-400">
            Data: {new Date(payload.updated_at).toLocaleTimeString()}
            {isStale && <span className="ml-2 text-amber-600 dark:text-amber-400">stale</span>}
            {lastFetchedAt > 0 && (
              <span className="ml-2">· polled {Math.round((Date.now() - lastFetchedAt) / 1000)}s ago</span>
            )}
          </div>
        )}
      </div>
      <p className="text-gray-600 dark:text-gray-400 mb-6">
        Live status of the <code className="text-xs bg-gray-100 dark:bg-gray-800 px-1 py-0.5 rounded">evo-reward-gpu</code> VM,
        GCS checkpoint state, and accumulated spend. Live estimate is computed from runtime × rate; billing-actual
        comes from the BigQuery billing export (~24h lag).
      </p>

      {fetchError && (
        <div className="mb-4 p-3 rounded-lg border border-red-300 dark:border-red-800 bg-red-50 dark:bg-red-950/30 text-sm text-red-700 dark:text-red-300">
          Failed to fetch status: {fetchError}. Showing last known data below.
        </div>
      )}

      {payload && payload.errors.length > 0 && (
        <div className="mb-4 p-3 rounded-lg border border-amber-300 dark:border-amber-800 bg-amber-50 dark:bg-amber-950/30 text-sm text-amber-800 dark:text-amber-300">
          <div className="font-medium mb-1">Partial data — {payload.errors.length} probe(s) failed:</div>
          <ul className="space-y-0.5 text-xs">
            {payload.errors.map((e, i) => (
              <li key={i} className="font-mono">{e.stage}: {e.message}</li>
            ))}
          </ul>
        </div>
      )}

      {payload && (
        <>
          <div className="grid gap-4 grid-cols-1 md:grid-cols-3">
            <VMCard vm={payload.vm} />
            <CheckpointsCard c={payload.checkpoints} />
            <CostsCard costs={payload.costs} />
          </div>

          {payload.training ? (
            <div className="mt-6">
              <TrainingCard t={payload.training} />
            </div>
          ) : (
            <div className="mt-6 p-4 rounded-lg border border-dashed border-gray-300 dark:border-gray-700 text-sm text-gray-500 dark:text-gray-400">
              No training progress reported yet. The runner writes <code className="text-xs bg-gray-100 dark:bg-gray-800 px-1 py-0.5 rounded">progress.json</code> every log interval; the gcs-sync sidecar pushes it to GCS every 5 min. Expect this card to appear within a few minutes of training starting.
            </div>
          )}

          <div className="mt-6 max-w-md">
            <ControlPanel vmStatus={payload.vm.status} />
          </div>
        </>
      )}

      {!payload && !fetchError && (
        <div className="text-gray-400 dark:text-gray-500">Loading...</div>
      )}
    </div>
  );
}
