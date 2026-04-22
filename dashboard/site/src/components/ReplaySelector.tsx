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
            (click a dot)
          </span>
        </div>
        <div className="relative h-10 border border-gray-200 dark:border-gray-800 rounded bg-gray-50 dark:bg-gray-900/50 px-2">
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
              <button
                key={r.path}
                onClick={() => onSelect(r)}
                className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 group"
                style={{ left: `calc(8px + ${frac} * (100% - 16px))` }}
                title={`step ${r.start_step.toLocaleString()} (+${r.n_frames})`}
              >
                <span
                  className={`block rounded-full transition ${
                    isSel
                      ? "w-3 h-3 bg-blue-600 ring-2 ring-blue-300 dark:ring-blue-900"
                      : "w-2 h-2 bg-gray-400 dark:bg-gray-500 hover:bg-blue-500"
                  }`}
                />
                <span className="absolute left-1/2 -translate-x-1/2 top-5 whitespace-nowrap text-[10px] font-mono text-gray-500 dark:text-gray-400 opacity-0 group-hover:opacity-100 pointer-events-none">
                  {r.start_step.toLocaleString()}
                </span>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
