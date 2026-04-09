import { BrowserRouter, Routes, Route, NavLink } from "react-router-dom";
import { lazy, Suspense, useState, useEffect } from "react";

// Pages — lazy loaded for code splitting
const ObservationLayout = lazy(() => import("./pages/ObservationLayout"));
const RewardEquation = lazy(() => import("./pages/RewardEquation"));
const AgentLifecycle = lazy(() => import("./pages/AgentLifecycle"));
const EvolutionVisualizer = lazy(() => import("./pages/EvolutionVisualizer"));
const SimLoopDiagram = lazy(() => import("./pages/SimLoopDiagram"));
const CodebaseMap = lazy(() => import("./pages/CodebaseMap"));
const ConfigReference = lazy(() => import("./pages/ConfigReference"));
const SessionTimeline = lazy(() => import("./pages/SessionTimeline"));
const DeviationTracker = lazy(() => import("./pages/DeviationTracker"));
const DocsViewer = lazy(() => import("./pages/DocsViewer"));

function useTheme() {
  const [dark, setDark] = useState(() => {
    if (typeof window === "undefined") return false;
    const stored = localStorage.getItem("evo-theme");
    if (stored) return stored === "dark";
    // Default to dark
    return true;
  });

  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
    localStorage.setItem("evo-theme", dark ? "dark" : "light");
  }, [dark]);

  return { dark, toggle: () => setDark((d) => !d) };
}

function ThemeToggle({ dark, toggle }: { dark: boolean; toggle: () => void }) {
  return (
    <button
      onClick={toggle}
      className="flex items-center gap-2 w-full px-3 py-2 rounded text-sm text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 transition"
      title={dark ? "Switch to light mode" : "Switch to dark mode"}
    >
      <span className="text-base">{dark ? "\u2600\uFE0F" : "\uD83C\uDF19"}</span>
      <span>{dark ? "Light mode" : "Dark mode"}</span>
    </button>
  );
}

function Home() {
  return (
    <div className="max-w-4xl mx-auto py-12 px-6">
      <h1 className="text-4xl font-bold mb-4 text-gray-900 dark:text-gray-100">evo-reward Explorer</h1>
      <p className="text-lg text-gray-600 dark:text-gray-400 mb-8">
        Interactive guide to the evolutionary reward structures project —
        replicating and extending Kanagawa & Doya (2025).
      </p>
      <div className="grid grid-cols-2 gap-4">
        {NAV_ITEMS.filter((n) => n.path !== "/").map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            className="block p-4 rounded-lg border border-gray-200 dark:border-gray-700 hover:border-blue-400 dark:hover:border-blue-500 hover:bg-blue-50 dark:hover:bg-blue-950/30 transition"
          >
            <h3 className="font-semibold text-lg text-gray-900 dark:text-gray-100">{item.label}</h3>
            <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">{item.description}</p>
          </NavLink>
        ))}
      </div>
    </div>
  );
}

function Loading() {
  return (
    <div className="max-w-4xl mx-auto py-12 px-6">
      <div className="text-gray-400 dark:text-gray-500">Loading...</div>
    </div>
  );
}

const NAV_ITEMS = [
  { path: "/", label: "Home", description: "Project overview" },
  { path: "/reward", label: "Reward Equation", description: "Slider-based reward explorer" },
  { path: "/observation", label: "Observation Layout", description: "Interactive 205-dim vector breakdown" },
  { path: "/sim-loop", label: "Simulation Loop", description: "10-step per-tick loop" },
  { path: "/evolution", label: "Evolution", description: "Mutation distribution visualizer" },
  { path: "/lifecycle", label: "Agent Lifecycle", description: "Birth-death-PPO flowchart" },
  { path: "/codebase", label: "Codebase Map", description: "Module dependency graph" },
  { path: "/config", label: "Config Reference", description: "Searchable parameter table" },
  { path: "/timeline", label: "Timeline", description: "Git commit project history" },
  { path: "/deviations", label: "Deviations", description: "Paper vs code tracker" },
  { path: "/docs", label: "Documentation", description: "Rendered markdown docs" },
];

export default function App() {
  const { dark, toggle } = useTheme();

  return (
    <BrowserRouter>
      <div className="min-h-screen bg-white dark:bg-gray-950 text-gray-900 dark:text-gray-100 transition-colors">
        {/* Sidebar */}
        <nav className="fixed left-0 top-0 h-full w-56 bg-gray-50 dark:bg-gray-900 border-r border-gray-200 dark:border-gray-800 p-4 overflow-y-auto flex flex-col">
          <div className="text-sm font-bold text-gray-400 dark:text-gray-500 uppercase tracking-wide mb-4">
            evo-reward
          </div>
          <div className="flex-1">
            {NAV_ITEMS.map((item) => (
              <NavLink
                key={item.path}
                to={item.path}
                end={item.path === "/"}
                className={({ isActive }) =>
                  `block px-3 py-2 rounded text-sm mb-1 transition ${
                    isActive
                      ? "bg-blue-100 dark:bg-blue-900/40 text-blue-800 dark:text-blue-300 font-medium"
                      : "text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800"
                  }`
                }
              >
                {item.label}
              </NavLink>
            ))}
          </div>
          <div className="pt-2 border-t border-gray-200 dark:border-gray-800">
            <ThemeToggle dark={dark} toggle={toggle} />
          </div>
        </nav>

        {/* Main content */}
        <main className="ml-56">
          <Suspense fallback={<Loading />}>
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/reward" element={<RewardEquation />} />
              <Route path="/observation" element={<ObservationLayout />} />
              <Route path="/sim-loop" element={<SimLoopDiagram />} />
              <Route path="/evolution" element={<EvolutionVisualizer />} />
              <Route path="/lifecycle" element={<AgentLifecycle />} />
              <Route path="/codebase" element={<CodebaseMap />} />
              <Route path="/config" element={<ConfigReference />} />
              <Route path="/timeline" element={<SessionTimeline />} />
              <Route path="/deviations" element={<DeviationTracker />} />
              <Route path="/docs" element={<DocsViewer />} />
            </Routes>
          </Suspense>
        </main>
      </div>
    </BrowserRouter>
  );
}
