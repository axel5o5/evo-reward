import { useMemo, useState } from "react";
import { ArchiveSummary, ReplayIndexEntry } from "../lib/replayLoader";
import {
  ExpCategory,
  ExpVariant,
  RunStats,
  categoryForExp,
  describeReplay,
  displayExperimentName,
  displayRunTag,
  formatSteps,
  parseRunTagDate,
  runStatsFor,
  tagStats,
  variantForExp,
} from "../lib/replayNaming";

interface Props {
  replays: ReplayIndexEntry[];
  selected: ReplayIndexEntry | null;
  onSelect: (entry: ReplayIndexEntry) => void;
  // Per-run rollup from summary.json. Optional so the selector still works
  // before the summary loads (or when the bucket has none yet).
  summary?: ArchiveSummary | null;
}

const tagOf = (r: ReplayIndexEntry) => r.run_tag || "current";

function pickFirst(
  replays: ReplayIndexEntry[],
  pred: (r: ReplayIndexEntry) => boolean,
): ReplayIndexEntry | undefined {
  return replays.find(pred);
}

function compareVariant(a: ExpVariant, b: ExpVariant): number {
  const rank = (v: ExpVariant): number => {
    if (v === "Baseline (paper)") return 0;
    if (v === "Baseline") return 1;
    if (v === "Axis 1") return 11;
    if (v === "Axis 2") return 12;
    if (v === "Axis 1+2") return 13;
    if (v === "Demo") return 90;
    return 50;
  };
  const ra = rank(a);
  const rb = rank(b);
  if (ra !== rb) return ra - rb;
  return a.localeCompare(b);
}

export default function ReplaySelector({
  replays,
  selected,
  onSelect,
  summary = null,
}: Props) {
  const [expSort, setExpSort] = useState<"recent" | "az">("recent");
  const fallback = replays[0] ?? null;
  const activeTag = selected ? tagOf(selected) : fallback ? tagOf(fallback) : "current";
  const activeVariant: ExpVariant = selected
    ? variantForExp(selected.exp)
    : fallback
      ? variantForExp(fallback.exp)
      : "Custom";
  const activeExp = selected?.exp ?? fallback?.exp ?? "";
  const activeSeed = selected?.seed ?? fallback?.seed ?? -1;
  const selectedDisplay = selected ? describeReplay(selected) : null;
  const selectedRunStats = selected
    ? runStatsFor(summary, selected.exp, selected.seed, tagOf(selected))
    : null;

  // Tags — "current" first, rest alpha.
  const tags = useMemo(() => {
    const s = new Set(replays.map(tagOf));
    return Array.from(s).sort((a, b) => {
      if (a === "current") return -1;
      if (b === "current") return 1;
      return a.localeCompare(b);
    });
  }, [replays]);

  // Tag → category, decided by the exps that show up under that tag. A tag
  // belongs to "active" iff any of its replays come from an active exp.
  // Tags map 1:1 to an exp in practice today, but the union semantics keep
  // the picker correct if that ever stops being true.
  const tagCategory = useMemo(() => {
    const m = new Map<string, ExpCategory>();
    for (const r of replays) {
      const t = tagOf(r);
      if (m.get(t) === "active") continue;
      m.set(t, categoryForExp(r.exp));
    }
    return m;
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
            {selectedRunStats && (
              <span
                className="px-2 py-0.5 rounded-full border border-gray-300 dark:border-gray-700 text-[11px] text-gray-600 dark:text-gray-300 font-mono"
                title={`Final step: ${selectedRunStats.finalStep.toLocaleString()}`}
              >
                {formatSteps(selectedRunStats.finalStep)} steps
              </span>
            )}
            {selectedRunStats && <ExtinctionBadge stats={selectedRunStats} />}
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
          tagCategory={tagCategory}
          summary={summary}
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
  tagCategory,
  summary,
  onSelect,
}: {
  tags: string[];
  activeTag: string;
  tagCategory: Map<string, ExpCategory>;
  summary: ArchiveSummary | null;
  onSelect: (tag: string) => void;
}) {
  // Top-level split mirrors configs/ vs configs/archive/. Active is the
  // collapsed-by-default exception inverted: open by default since that's
  // where the in-flight runs live; archive starts closed but auto-opens
  // if the current selection lives there.
  const activeTags = tags.filter((t) => (tagCategory.get(t) ?? "archive") === "active");
  const archiveTags = tags.filter((t) => (tagCategory.get(t) ?? "archive") === "archive");
  const activeCategory: ExpCategory = (tagCategory.get(activeTag) ?? "archive");

  return (
    <div>
      <div className="text-[10px] font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-1">
        Run
      </div>
      <div className="flex flex-col gap-1">
        <CategorySection
          label="Active"
          hint="from configs/"
          tags={activeTags}
          activeTag={activeTag}
          summary={summary}
          onSelect={onSelect}
          defaultOpen={true}
          forceOpen={activeCategory === "active"}
        />
        <CategorySection
          label="Archive"
          hint="from configs/archive/"
          tags={archiveTags}
          activeTag={activeTag}
          summary={summary}
          onSelect={onSelect}
          defaultOpen={false}
          forceOpen={activeCategory === "archive"}
        />
      </div>
    </div>
  );
}

// One collapsible group (Active or Archive). Inside, the existing date bands
// still apply — recent ones rendered inline, older ones behind a Show toggle.
function CategorySection({
  label,
  hint,
  tags,
  activeTag,
  summary,
  onSelect,
  defaultOpen,
  forceOpen,
}: {
  label: string;
  hint: string;
  tags: string[];
  activeTag: string;
  summary: ArchiveSummary | null;
  onSelect: (tag: string) => void;
  defaultOpen: boolean;
  forceOpen: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
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
    for (const k of Object.keys(buckets) as DateBand[]) {
      buckets[k].sort((a, b) => {
        if (a === "current") return -1;
        if (b === "current") return 1;
        return b.localeCompare(a);
      });
    }
    return buckets;
  }, [tags]);

  const activeIsOlder = useMemo(
    () => OLDER_BANDS.some((b) => banded[b].includes(activeTag)),
    [banded, activeTag],
  );
  const olderOpen = showOlder || activeIsOlder;
  const olderCount = OLDER_BANDS.reduce((n, b) => n + banded[b].length, 0);

  if (tags.length === 0) return null;
  const isOpen = open || forceOpen;

  return (
    <div className="rounded border border-gray-200 dark:border-gray-800">
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center justify-between gap-2 px-2 py-1.5 text-left hover:bg-gray-50 dark:hover:bg-gray-900/40"
        aria-expanded={isOpen}
      >
        <span className="flex items-baseline gap-2">
          <span className="text-xs font-semibold text-gray-800 dark:text-gray-200">
            {label}
          </span>
          <span className="text-[10px] text-gray-400 dark:text-gray-500">
            {hint}
          </span>
        </span>
        <span className="flex items-center gap-1.5">
          <span className="text-[10px] tabular-nums text-gray-500 dark:text-gray-400">
            {tags.length}
          </span>
          <span className="text-gray-400 text-xs">{isOpen ? "▾" : "▸"}</span>
        </span>
      </button>
      {isOpen && (
        <div className="px-2 pb-2 pt-1 flex flex-col gap-1">
          {RECENT_BANDS.map((band) =>
            banded[band].length > 0 ? (
              <RunBand
                key={band}
                label={band}
                tags={banded[band]}
                activeTag={activeTag}
                summary={summary}
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
                  summary={summary}
                  onSelect={onSelect}
                />
              ) : null,
            )}
        </div>
      )}
    </div>
  );
}

function RunBand({
  label,
  tags,
  activeTag,
  summary,
  onSelect,
}: {
  label: DateBand;
  tags: string[];
  activeTag: string;
  summary: ArchiveSummary | null;
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
            stats={tagStats(summary, t)}
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
  stats,
  onSelect,
}: {
  tag: string;
  active: boolean;
  stats: RunStats | null;
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
  const titleParts = [tag];
  if (stats) {
    titleParts.push(`${stats.finalStep.toLocaleString()} steps`);
    if (stats.extinct) {
      const sp = stats.extinctSpecies;
      const at =
        stats.extinctionStep !== null
          ? ` @ ${stats.extinctionStep.toLocaleString()}`
          : "";
      titleParts.push(`${sp} extinct${at}`);
    }
  }
  return (
    <button
      onClick={onSelect}
      className={`flex items-baseline gap-2 px-2 py-1 rounded text-left text-xs transition ${
        active
          ? "bg-blue-600 text-white"
          : "hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-700 dark:text-gray-300"
      }`}
      title={titleParts.join(" · ")}
    >
      <span
        className={`font-mono text-[10px] tabular-nums ${
          active ? "text-blue-100" : "text-gray-400 dark:text-gray-500"
        }`}
      >
        {dateLabel}
      </span>
      <span className="truncate flex-1">{label}</span>
      {stats && (
        <span
          className={`font-mono text-[10px] tabular-nums ${
            active ? "text-blue-100" : "text-gray-500 dark:text-gray-400"
          }`}
        >
          {formatSteps(stats.finalStep)}
        </span>
      )}
      {stats?.extinct && <ExtinctionBadge stats={stats} compact active={active} />}
    </button>
  );
}

// Extinction badge — red for predator extinction, amber for prey, slate for
// mixed. `compact` is the small inline variant used inside RunRow; the
// non-compact form is the chip-shaped one in the Selected card.
function ExtinctionBadge({
  stats,
  compact = false,
  active = false,
}: {
  stats: RunStats;
  compact?: boolean;
  active?: boolean;
}) {
  if (!stats.extinct) return null;
  const sp = stats.extinctSpecies;
  const label = sp === "pred" ? "✗ pred" : sp === "prey" ? "✗ prey" : "✗ mixed";
  const titleAt =
    stats.extinctionStep !== null
      ? ` at step ${stats.extinctionStep.toLocaleString()}`
      : "";
  const title =
    sp === "mixed"
      ? `Multiple seeds extincted in different species${titleAt}`
      : `${sp === "pred" ? "Predator" : "Prey"} went extinct${titleAt}`;
  // Active-row variant rides on the blue background — keep contrast with a
  // light fill instead of trying to color-shift the underlying button.
  if (active) {
    return (
      <span
        title={title}
        className="font-mono text-[10px] tabular-nums px-1 rounded bg-blue-100 text-blue-700"
      >
        {label}
      </span>
    );
  }
  const cls =
    sp === "pred"
      ? "border-red-400 text-red-700 dark:border-red-700 dark:text-red-300 bg-red-50 dark:bg-red-950/40"
      : sp === "prey"
        ? "border-amber-400 text-amber-700 dark:border-amber-700 dark:text-amber-300 bg-amber-50 dark:bg-amber-950/40"
        : "border-gray-400 text-gray-700 dark:border-gray-600 dark:text-gray-300 bg-gray-50 dark:bg-gray-900";
  if (compact) {
    return (
      <span
        title={title}
        className={`font-mono text-[10px] tabular-nums px-1 rounded border ${cls}`}
      >
        {label}
      </span>
    );
  }
  return (
    <span
      title={title}
      className={`px-2 py-0.5 rounded-full border text-[11px] font-mono ${cls}`}
    >
      {label}
    </span>
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
