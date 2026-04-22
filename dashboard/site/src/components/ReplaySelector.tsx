import { useMemo } from "react";
import { ReplayIndexEntry } from "../lib/replayLoader";

interface Props {
  replays: ReplayIndexEntry[];
  selected: ReplayIndexEntry | null;
  onSelect: (entry: ReplayIndexEntry) => void;
}

const tagOf = (r: ReplayIndexEntry) => r.run_tag || "current";

const tagLabel = (tag: string) =>
  tag === "current" ? "Current" : tag.replace(/_/g, " ");

function pickFirst(
  replays: ReplayIndexEntry[],
  pred: (r: ReplayIndexEntry) => boolean,
): ReplayIndexEntry | undefined {
  return replays.find(pred);
}

export default function ReplaySelector({ replays, selected, onSelect }: Props) {
  const activeTag = selected ? tagOf(selected) : "current";
  const activeExp = selected?.exp ?? "";
  const activeSeed = selected?.seed ?? -1;

  // Tags — "current" first, rest alpha.
  const tags = useMemo(() => {
    const s = new Set(replays.map(tagOf));
    return Array.from(s).sort((a, b) => {
      if (a === "current") return -1;
      if (b === "current") return 1;
      return a.localeCompare(b);
    });
  }, [replays]);

  const expsInTag = useMemo(() => {
    const s = new Set(replays.filter((r) => tagOf(r) === activeTag).map((r) => r.exp));
    return Array.from(s).sort();
  }, [replays, activeTag]);

  const seedsInExp = useMemo(() => {
    const s = new Set(
      replays
        .filter((r) => tagOf(r) === activeTag && r.exp === activeExp)
        .map((r) => r.seed),
    );
    return Array.from(s).sort((a, b) => a - b);
  }, [replays, activeTag, activeExp]);

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
      {tags.length > 1 && (
        <Row label="Run">
          {tags.map((t) => (
            <button
              key={t}
              onClick={() => pickTag(t)}
              className={chipClass(t === activeTag)}
              title={t}
            >
              {tagLabel(t)}
            </button>
          ))}
        </Row>
      )}

      <Row label="Experiment">
        {expsInTag.map((e) => (
          <button
            key={e}
            onClick={() => pickExp(e)}
            className={chipClass(e === activeExp)}
          >
            {e}
          </button>
        ))}
      </Row>

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
  const title = `${entry.exp} seed ${entry.seed} · step ${entry.start_step.toLocaleString()} (+${entry.n_frames})`;
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
