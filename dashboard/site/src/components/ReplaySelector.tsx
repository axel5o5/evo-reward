import { useMemo, useState } from "react";
import { ReplayIndexEntry } from "../lib/replayLoader";
import {
  describeReplay,
  displayExperimentName,
  displayRunTag,
  parseRunTagDate,
  variantForExp,
} from "../lib/replayNaming";

interface Props {
  replays: ReplayIndexEntry[];
  selected: ReplayIndexEntry | null;
  onSelect: (entry: ReplayIndexEntry) => void;
}

const tagOf = (r: ReplayIndexEntry) => r.run_tag || "current";

function pickFirst(
  replays: ReplayIndexEntry[],
  pred: (r: ReplayIndexEntry) => boolean,
): ReplayIndexEntry | undefined {
  return replays.find(pred);
}

function compareVariant(a: string, b: string): number {
  const rank = (v: string): number => {
    if (v === "Baseline") return 0;
    const axis = v.match(/^Axis (\d+)$/);
    if (axis) return 10 + Number(axis[1]);
    if (v === "Demo") return 90;
    return 50;
  };
  const ra = rank(a);
  const rb = rank(b);
  if (ra !== rb) return ra - rb;
  return a.localeCompare(b);
}

export default function ReplaySelector({ replays, selected, onSelect }: Props) {
  const [expSort, setExpSort] = useState<"recent" | "az">("recent");
  const fallback = replays[0] ?? null;
  const activeTag = selected ? tagOf(selected) : fallback ? tagOf(fallback) : "current";
  const activeVariant = selected
    ? variantForExp(selected.exp)
    : fallback
      ? variantForExp(fallback.exp)
      : "";
  const activeExp = selected?.exp ?? fallback?.exp ?? "";
  const activeSeed = selected?.seed ?? fallback?.seed ?? -1;
  const selectedDisplay = selected ? describeReplay(selected) : null;

  // Tags — "current" first, rest alpha.
  const tags = useMemo(() => {
    const s = new Set(replays.map(tagOf));
    return Array.from(s).sort((a, b) => {
      if (a === "current") return -1;
      if (b === "current") return 1;
      return a.localeCompare(b);
    });
  }, [replays]);

  const variantsInTag = useMemo(() => {
    const s = new Set(
      replays
        .filter((r) => tagOf(r) === activeTag)
        .map((r) => variantForExp(r.exp)),
    );
    return Array.from(s).sort(compareVariant);
  }, [replays, activeTag]);

  const expsInVariant = useMemo(() => {
    const grouped = new Map<string, number>();
    for (const r of replays) {
      if (tagOf(r) !== activeTag) continue;
      if (variantForExp(r.exp) !== activeVariant) continue;
      const cur = grouped.get(r.exp) ?? -1;
      if (r.start_step > cur) grouped.set(r.exp, r.start_step);
    }
    const entries = Array.from(grouped.entries());
    if (expSort === "recent") {
      entries.sort((a, b) => b[1] - a[1]);
    } else {
      entries.sort((a, b) =>
        displayExperimentName(a[0]).localeCompare(displayExperimentName(b[0])),
      );
    }
    return entries.map(([exp]) => exp);
  }, [replays, activeTag, activeVariant, expSort]);

  const seedsInExp = useMemo(() => {
    const s = new Set(
      replays
        .filter(
          (r) =>
            tagOf(r) === activeTag &&
            variantForExp(r.exp) === activeVariant &&
            r.exp === activeExp,
        )
        .map((r) => r.seed),
    );
    return Array.from(s).sort((a, b) => a - b);
  }, [replays, activeTag, activeVariant, activeExp]);

  const timeline = useMemo(() => {
    const matches = replays
      .filter(
        (r) =>
          tagOf(r) === activeTag &&
          r.exp === activeExp &&
          r.seed === activeSeed,
      )
      .sort((a, b) => a.start_step - b.start_step);
    const maxStep = matches.reduce(
      (m, r) => Math.max(m, r.start_step + r.n_frames),
      0,
    );
    return { matches, maxStep };
  }, [replays, activeTag, activeExp, activeSeed]);

  const pickTag = (tag: string) => {
    const next =
      pickFirst(replays, (r) => tagOf(r) === tag && r.exp === activeExp && r.seed === activeSeed) ??
      pickFirst(replays, (r) => tagOf(r) === tag && r.exp === activeExp) ??
      pickFirst(replays, (r) => tagOf(r) === tag);
    if (next) onSelect(next);
  };
  const pickExp = (exp: string) => {
    const next =
      pickFirst(
        replays,
        (r) => tagOf(r) === activeTag && r.exp === exp && r.seed === activeSeed,
      ) ?? pickFirst(replays, (r) => tagOf(r) === activeTag && r.exp === exp);
    if (next) onSelect(next);
  };
  const pickVariant = (variant: string) => {
    const next =
      pickFirst(
        replays,
        (r) =>
          tagOf(r) === activeTag &&
          variantForExp(r.exp) === variant &&
          r.exp === activeExp &&
          r.seed === activeSeed,
      ) ??
      pickFirst(
        replays,
        (r) =>
          tagOf(r) === activeTag &&
          variantForExp(r.exp) === variant &&
          r.exp === activeExp,
      ) ??
      pickFirst(
        replays,
        (r) => tagOf(r) === activeTag && variantForExp(r.exp) === variant,
      );
    if (next) onSelect(next);
  };
  const pickSeed = (seed: number) => {
    const next = pickFirst(
      replays,
      (r) => tagOf(r) === activeTag && r.exp === activeExp && r.seed === seed,
    );
    if (next) onSelect(next);
  };

  const Row = ({
    label,
    children,
  }: {
    label: string;
    children: React.ReactNode;
  }) => (
    <div>
      <div className="text-[10px] font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-1">
        {label}
      </div>
      <div className="flex flex-wrap gap-1.5">{children}</div>
    </div>
  );

  const chipClass = (active: boolean) =>
    `px-2 py-1 rounded-full text-xs font-medium border transition ${
      active
        ? "bg-blue-600 border-blue-600 text-white"
        : "bg-white dark:bg-gray-900 border-gray-300 dark:border-gray-700 text-gray-700 dark:text-gray-300 hover:border-blue-400 dark:hover:border-blue-500"
    }`;

  return (
    <div className="flex flex-col gap-3">
      {selectedDisplay && (
        <div className="rounded-lg border border-gray-200 dark:border-gray-800 bg-gray-50 dark:bg-gray-900/50 p-3">
          <div className="text-[10px] font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400">
            Selected replay
          </div>
          <div className="mt-1 text-sm font-semibold text-gray-900 dark:text-gray-100">
            {selectedDisplay.title}
          </div>
          <div className="mt-2 flex flex-wrap gap-1.5">
            {selectedDisplay.chips.map((chip) => (
              <span
                key={chip}
                className="px-2 py-0.5 rounded-full border border-gray-300 dark:border-gray-700 text-[11px] text-gray-600 dark:text-gray-300"
              >
                {chip}
              </span>
            ))}
          </div>
          <div className="mt-2 text-[10px] text-gray-500 dark:text-gray-400">
            Run: {selectedDisplay.runLabel}
          </div>
          <div className="mt-0.5 text-[10px] font-mono text-gray-500 dark:text-gray-400 break-all">
            Raw id: {selectedDisplay.rawId}
          </div>
        </div>
      )}

      {tags.length > 1 && (
        <RunPicker
          tags={tags}
          activeTag={activeTag}
          onSelect={pickTag}
        />
      )}

      {variantsInTag.length > 1 && (
        <Row label="Variant">
          {variantsInTag.map((v) => (
            <button
              key={v}
              onClick={() => pickVariant(v)}
              className={chipClass(v === activeVariant)}
            >
              {v}
            </button>
          ))}
        </Row>
      )}

      <div>
        <div className="text-[10px] font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-1 flex items-center justify-between gap-2">
          <span>Simulation</span>
          <select
            value={expSort}
            onChange={(e) => setExpSort(e.target.value as "recent" | "az")}
            className="text-[11px] normal-case tracking-normal border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-900 rounded px-1.5 py-0.5 text-gray-600 dark:text-gray-300"
            title="Sort simulations"
          >
            <option value="recent">Most recent</option>
            <option value="az">A → Z</option>
          </select>
        </div>
        <div className="flex flex-col gap-1.5">
          {expsInVariant.map((e) => {
            const active = e === activeExp;
            return (
              <button
                key={e}
                onClick={() => pickExp(e)}
                className={`w-full rounded-lg border p-2 text-left transition ${
                  active
                    ? "border-blue-600 bg-blue-50 dark:bg-blue-950/40"
                    : "border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-900 hover:border-blue-400 dark:hover:border-blue-500"
                }`}
              >
                <div className="text-sm font-medium text-gray-900 dark:text-gray-100">
                  {displayExperimentName(e)}
                </div>
                <div className="text-[10px] font-mono text-gray-500 dark:text-gray-400 mt-0.5">
                  {e}
                </div>
              </button>
            );
          })}
        </div>
      </div>

      <Row label="Seed">
        {seedsInExp.map((s) => (
          <button
            key={s}
            onClick={() => pickSeed(s)}
            className={chipClass(s === activeSeed)}
          >
            {s}
          </button>
        ))}
      </Row>

      <div>
        <div className="text-[10px] font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-1">
          Start step{" "}
          <span className="text-gray-400 dark:text-gray-500 normal-case tracking-normal">
            (click a marker)
          </span>
        </div>
        <div className="relative h-12 border border-gray-200 dark:border-gray-800 rounded bg-gray-50 dark:bg-gray-900/50 px-2">
          {/* baseline */}
          <div className="absolute left-2 right-2 top-1/2 h-px bg-gray-300 dark:bg-gray-700" />
          {timeline.matches.map((r) => {
            const frac =
              timeline.maxStep > 0 ? r.start_step / timeline.maxStep : 0;
            const isSel =
              selected &&
              tagOf(selected) === tagOf(r) &&
              selected.exp === r.exp &&
              selected.seed === r.seed &&
              selected.start_step === r.start_step;
            return (
              <TimelineMarker
                key={r.path}
                entry={r}
                frac={frac}
                selected={!!isSel}
                onSelect={() => onSelect(r)}
              />
            );
          })}
        </div>
      </div>
    </div>
  );
}

// Date-banded run picker. Replaces the previous wrap-on-overflow chip row
// because tag length varies (e.g. `2026-04-22T2328Z_phase1a-v7-seed1-sensor120`
// vs `2026-04-21`), which produced ugly inconsistent wrapping. Vertical list
// + date prefix column gives consistent layout AND makes "what's old vs new"
// obvious at a glance.
type DateBand =
  | "today"
  | "yesterday"
  | "this week"
  | "this month"
  | "older"
  | "undated";

const RECENT_BANDS: DateBand[] = ["today", "yesterday", "this week"];
const OLDER_BANDS: DateBand[] = ["this month", "older", "undated"];

function bandOf(date: Date | null, now: Date): DateBand {
  if (!date) return "undated";
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const days = Math.floor((startOfToday.getTime() - date.getTime()) / 86400000);
  if (days <= 0) return "today";
  if (days === 1) return "yesterday";
  if (days <= 7) return "this week";
  if (days <= 30) return "this month";
  return "older";
}

function RunPicker({
  tags,
  activeTag,
  onSelect,
}: {
  tags: string[];
  activeTag: string;
  onSelect: (tag: string) => void;
}) {
  const [showOlder, setShowOlder] = useState(false);

  const banded = useMemo(() => {
    const now = new Date();
    const buckets: Record<DateBand, string[]> = {
      today: [], yesterday: [], "this week": [],
      "this month": [], older: [], undated: [],
    };
    for (const t of tags) {
      const date = t === "current" ? null : parseRunTagDate(t);
      buckets[bandOf(date, now)].push(t);
    }
    // Within a band: most recent first (lexical works because date prefix is ISO).
    for (const k of Object.keys(buckets) as DateBand[]) {
      buckets[k].sort((a, b) => {
        if (a === "current") return -1;
        if (b === "current") return 1;
        return b.localeCompare(a);
      });
    }
    return buckets;
  }, [tags]);

  // If the active tag falls in an "older" band, auto-expand so the user sees
  // their selection rather than a hidden state.
  const activeIsOlder = useMemo(
    () => OLDER_BANDS.some((b) => banded[b].includes(activeTag)),
    [banded, activeTag],
  );
  const olderOpen = showOlder || activeIsOlder;

  const olderCount = OLDER_BANDS.reduce((n, b) => n + banded[b].length, 0);

  return (
    <div>
      <div className="text-[10px] font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-1">
        Run
      </div>
      <div className="flex flex-col gap-1">
        {RECENT_BANDS.map((band) =>
          banded[band].length > 0 ? (
            <RunBand
              key={band}
              label={band}
              tags={banded[band]}
              activeTag={activeTag}
              onSelect={onSelect}
            />
          ) : null,
        )}

        {olderCount > 0 && (
          <button
            onClick={() => setShowOlder((v) => !v)}
            className="text-[10px] uppercase tracking-wider text-blue-600 dark:text-blue-400 hover:underline self-start mt-1"
          >
            {olderOpen ? "▾ Hide" : "▸ Show"} older ({olderCount})
          </button>
        )}

        {olderOpen &&
          OLDER_BANDS.map((band) =>
            banded[band].length > 0 ? (
              <RunBand
                key={band}
                label={band}
                tags={banded[band]}
                activeTag={activeTag}
                onSelect={onSelect}
              />
            ) : null,
          )}
      </div>
    </div>
  );
}

function RunBand({
  label,
  tags,
  activeTag,
  onSelect,
}: {
  label: DateBand;
  tags: string[];
  activeTag: string;
  onSelect: (tag: string) => void;
}) {
  return (
    <div>
      <div className="text-[9px] uppercase tracking-wider text-gray-400 dark:text-gray-500 mb-0.5 mt-1">
        {label}
      </div>
      <div className="flex flex-col">
        {tags.map((t) => (
          <RunRow
            key={t}
            tag={t}
            active={t === activeTag}
            onSelect={() => onSelect(t)}
          />
        ))}
      </div>
    </div>
  );
}

function RunRow({
  tag,
  active,
  onSelect,
}: {
  tag: string;
  active: boolean;
  onSelect: () => void;
}) {
  const date = tag === "current" ? null : parseRunTagDate(tag);
  const dateLabel = date
    ? `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}-${String(date.getDate()).padStart(2, "0")}`
    : "—";
  // displayRunTag returns "Apr 21 · D19", "Apr 21", or just "Current". Since
  // we render the date in its own column, strip the "Mon DD" prefix from the
  // label. Date-only tags (no descriptor) fall through to an em-dash.
  const display = displayRunTag(tag === "current" ? undefined : tag);
  const stripped = display.replace(/^[A-Z][a-z]{2} \d{1,2}(?: · )?/, "");
  const label = stripped || (date ? "—" : display);
  return (
    <button
      onClick={onSelect}
      className={`flex items-baseline gap-2 px-2 py-1 rounded text-left text-xs transition ${
        active
          ? "bg-blue-600 text-white"
          : "hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-700 dark:text-gray-300"
      }`}
      title={tag}
    >
      <span
        className={`font-mono text-[10px] tabular-nums ${
          active ? "text-blue-100" : "text-gray-400 dark:text-gray-500"
        }`}
      >
        {dateLabel}
      </span>
      <span className="truncate flex-1">{label}</span>
    </button>
  );
}

// A marker on the start-step timeline. Renders a 60×18 sparkline thumbnail
// if the entry has one; otherwise a plain dot.
function TimelineMarker({
  entry,
  frac,
  selected,
  onSelect,
}: {
  entry: ReplayIndexEntry;
  frac: number;
  selected: boolean;
  onSelect: () => void;
}) {
  const spark = entry.sparkline;
  const display = describeReplay(entry);
  const title = `${display.title} · ${display.runLabel} · seed ${entry.seed} · step ${entry.start_step.toLocaleString()} (+${entry.n_frames})`;
  return (
    <button
      onClick={onSelect}
      className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 group"
      style={{ left: `calc(8px + ${frac} * (100% - 16px))` }}
      title={title}
    >
      {spark ? (
        <SparklineThumb
          prey={spark.prey}
          pred={spark.pred}
          selected={selected}
        />
      ) : (
        <span
          className={`block rounded-full transition ${
            selected
              ? "w-3 h-3 bg-blue-600 ring-2 ring-blue-300 dark:ring-blue-900"
              : "w-2 h-2 bg-gray-400 dark:bg-gray-500 hover:bg-blue-500"
          }`}
        />
      )}
      <span className="absolute left-1/2 -translate-x-1/2 top-[22px] whitespace-nowrap text-[10px] font-mono text-gray-500 dark:text-gray-400 opacity-0 group-hover:opacity-100 pointer-events-none">
        {entry.start_step.toLocaleString()}
      </span>
    </button>
  );
}

const SPARK_W = 60;
const SPARK_H = 18;

function SparklineThumb({
  prey,
  pred,
  selected,
}: {
  prey: number[];
  pred: number[];
  selected: boolean;
}) {
  const n = Math.max(prey.length, pred.length);
  let maxC = 1;
  for (const v of prey) if (v > maxC) maxC = v;
  for (const v of pred) if (v > maxC) maxC = v;
  const pts = (arr: number[]) =>
    arr
      .map((v, i) => {
        const x = n > 1 ? (i / (n - 1)) * SPARK_W : 0;
        const y = SPARK_H - (v / maxC) * SPARK_H;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");
  return (
    <svg
      width={SPARK_W}
      height={SPARK_H}
      viewBox={`0 0 ${SPARK_W} ${SPARK_H}`}
      preserveAspectRatio="none"
      className={`block rounded border transition ${
        selected
          ? "border-blue-500 bg-blue-50 dark:bg-blue-950/40"
          : "border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-900 hover:border-blue-400"
      }`}
    >
      <polyline
        points={pts(prey)}
        fill="none"
        stroke="#4ade80"
        strokeWidth={1}
        vectorEffect="non-scaling-stroke"
      />
      <polyline
        points={pts(pred)}
        fill="none"
        stroke="#f87171"
        strokeWidth={1}
        vectorEffect="non-scaling-stroke"
      />
    </svg>
  );
}
