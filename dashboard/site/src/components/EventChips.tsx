import { ReplayStats, nextEventAfter } from "../lib/replayStats";

interface Props {
  stats: ReplayStats;
  frameIdx: number;
  onJump: (frame: number) => void;
}

// "Interesting frames" chip row. Clicking a chip jumps to the next
// occurrence after frameIdx (wraps to the first occurrence).
// Peak/crossover chips jump to the single known frame.
export default function EventChips({ stats, frameIdx, onJump }: Props) {
  const births = stats.births.length;
  const deaths = stats.deaths.length;

  const chip = (
    label: string,
    target: number,
    disabled: boolean,
    title?: string,
  ) => (
    <button
      onClick={() => {
        if (!disabled && target >= 0) onJump(target);
      }}
      disabled={disabled || target < 0}
      title={title}
      className="px-2 py-0.5 rounded-full text-[11px] font-mono border border-gray-300 dark:border-gray-700 text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-900 hover:border-blue-400 hover:text-blue-700 dark:hover:text-blue-300 disabled:opacity-40 disabled:cursor-not-allowed transition"
    >
      {label}
    </button>
  );

  return (
    <div className="flex flex-wrap items-center gap-1.5">
      <span className="text-[10px] uppercase tracking-wider text-gray-500 dark:text-gray-400 mr-1">
        jump
      </span>
      {chip(
        `▶ birth ×${births}`,
        nextEventAfter(stats.births, frameIdx),
        births === 0,
        "Next frame with at least one birth (wraps around)",
      )}
      {chip(
        `☠ death ×${deaths}`,
        nextEventAfter(stats.deaths, frameIdx),
        deaths === 0,
        "Next frame with at least one death (wraps around)",
      )}
      {chip(
        `↑ peak prey @${stats.peakPreyFrame}`,
        stats.peakPreyFrame,
        false,
        "Frame with maximum live prey",
      )}
      {chip(
        `↑ peak pred @${stats.peakPredFrame}`,
        stats.peakPredFrame,
        false,
        "Frame with maximum live predators",
      )}
      {stats.firstCrossoverFrame >= 0
        ? chip(
            `⇄ crossover @${stats.firstCrossoverFrame}`,
            stats.firstCrossoverFrame,
            false,
            "First frame where pred/prey dominance flips vs. frame 0",
          )
        : chip("⇄ no crossover", -1, true)}
    </div>
  );
}
