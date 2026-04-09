import { useState } from "react";

type Paper = {
  id: string;
  authors: string;
  year: number;
  title: string;
  venue: string;
  category: string;
  relevance: string;
  url?: string;
};

const PAPERS: Paper[] = [
  {
    id: "kanagawa2025",
    authors: "Kanagawa, Y., & Doya, K.",
    year: 2025,
    title: "Evolution of fear and social rewards in prey-predator relationship",
    venue: "arXiv:2507.09992 (under review)",
    category: "Core",
    relevance: "The paper being replicated. Prey evolve fear (negative predator reward) and social affiliation via birth-death selection on reward genomes. Inner loop: PPO. Outer loop: natural selection.",
    url: "https://arxiv.org/abs/2507.09992",
  },
  {
    id: "kanagawa2024",
    authors: "Kanagawa, Y., & Doya, K.",
    year: 2024,
    title: "Evolution of rewards for food and motor action by simulating birth and death",
    venue: "ALIFE 2024, MIT Press",
    category: "Core",
    relevance: "Direct precursor. Introduces the core simulation: agents evolve what to care about, then learn behaviors via RL. Demonstrates food-seeking rewards emerge from birth-death selection.",
  },
  {
    id: "doya2002",
    authors: "Doya, K.",
    year: 2002,
    title: "Metalearning and neuromodulation",
    venue: "Neural Networks, 15(4-6), 495-506",
    category: "Core",
    relevance: "Theoretical anchor. Proposes that neuromodulatory systems implement RL meta-parameters: dopamine = TD error, serotonin = time horizon, acetylcholine = learning rate.",
  },
  {
    id: "du2019",
    authors: "Du, Y., Han, L., Fang, M., Liu, J., Jiang, T., & Tao, D.",
    year: 2019,
    title: "LIIR: Learning individual intrinsic reward in multi-agent reinforcement learning",
    venue: "NeurIPS 2019",
    category: "Intrinsic Motivation",
    relevance: "Closest multi-agent precedent. Each agent learns its own intrinsic reward via bi-level optimization. Key difference: LIIR is cooperative, our setting is competitive.",
    url: "https://arxiv.org/abs/1907.07115",
  },
  {
    id: "pathak2017",
    authors: "Pathak, D., Agrawal, P., Efros, A. A., & Darrell, T.",
    year: 2017,
    title: "Curiosity-driven exploration by self-supervised prediction",
    venue: "ICML 2017",
    category: "Intrinsic Motivation",
    relevance: "Canonical reference for learned intrinsic reward (ICM). Fixed architecture curiosity — agents don't discover what kind of thing to be curious about.",
    url: "https://arxiv.org/abs/1705.05363",
  },
  {
    id: "burda2019",
    authors: "Burda, Y., Edwards, H., Storkey, A., & Klimov, O.",
    year: 2019,
    title: "Exploration by random network distillation",
    venue: "ICLR 2019",
    category: "Intrinsic Motivation",
    relevance: "RND. Simpler than ICM, avoids the noisy TV problem. Recommended fixed intrinsic reward baseline.",
    url: "https://arxiv.org/abs/1810.12894",
  },
  {
    id: "bansal2018",
    authors: "Bansal, T., Pachocki, J., Sidor, S., Sutskever, I., & Mordatch, I.",
    year: 2018,
    title: "Emergent complexity via multi-agent competition",
    venue: "ICLR 2018",
    category: "Multi-Agent RL",
    relevance: "Demonstrates competitive self-play produces behavioral complexity exceeding task reward. Complexity scales with opponent difficulty.",
    url: "https://arxiv.org/abs/1710.03748",
  },
  {
    id: "baker2020",
    authors: "Baker, B., et al.",
    year: 2020,
    title: "Emergent tool use from multi-agent autocurricula",
    venue: "ICLR 2020",
    category: "Multi-Agent RL",
    relevance: "Hide-and-seek. Six strategy phases emerge from simple reward. Established autocurriculum concept and behavioral phase analysis methodology.",
    url: "https://arxiv.org/abs/1909.07528",
  },
  {
    id: "jaderberg2019",
    authors: "Jaderberg, M., et al.",
    year: 2019,
    title: "Human-level performance in 3D multiplayer games with population-based reinforcement learning",
    venue: "Science, 364(6443), 859-865",
    category: "Multi-Agent RL",
    relevance: "FTW / Quake III CTF. Inner loop PPO, outer loop PBT adapting internal reward parameters. Learned internal rewards produce qualitatively different behaviors than fixed rewards.",
  },
  {
    id: "schulman2017",
    authors: "Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O.",
    year: 2017,
    title: "Proximal policy optimization algorithms",
    venue: "arXiv:1707.06347",
    category: "RL Algorithms",
    relevance: "PPO. The inner-loop algorithm for all agent policies in this project. Clipped surrogate objective, simple and stable.",
    url: "https://arxiv.org/abs/1707.06347",
  },
  {
    id: "lowe2017",
    authors: "Lowe, R., Wu, Y., Tamar, A., Harb, J., Abbeel, P., & Mordatch, I.",
    year: 2017,
    title: "Multi-agent actor-critic for mixed cooperative-competitive environments",
    venue: "NeurIPS 2017",
    category: "RL Algorithms",
    relevance: "MADDPG. Centralized training, decentralized execution. Addresses non-stationarity in competitive MARL.",
    url: "https://arxiv.org/abs/1706.02275",
  },
  {
    id: "christianos2021",
    authors: "Christianos, F., Schafer, L., & Albrecht, S.",
    year: 2021,
    title: "Scaling multi-agent reinforcement learning with selective parameter sharing",
    venue: "NeurIPS 2021",
    category: "Multi-Agent RL",
    relevance: "When does sharing help vs. hurt in heterogeneous MARL? Our agents are genetically heterogeneous, making this a relevant concern for shared policy mode.",
  },
  {
    id: "bredeche2012",
    authors: "Bredeche, N., & Montanier, J.-M.",
    year: 2012,
    title: "mEDEA: Embodied evolution with implicit fitness",
    venue: "Robotics and Autonomous Systems",
    category: "Evolutionary",
    relevance: "Closest analog to K&D's continuous birth-death in the robotics literature. Implicit fitness from embodied evolution.",
  },
  {
    id: "stanley2002",
    authors: "Stanley, K. O., & Miikkulainen, R.",
    year: 2002,
    title: "Evolving neural networks through augmenting topologies",
    venue: "Evolutionary Computation, 10(2), 99-127",
    category: "Evolutionary",
    relevance: "NEAT. Evolves network weights and topology. Historical markings, speciation, minimal starting structure.",
  },
  {
    id: "khadka2018",
    authors: "Khadka, S., & Tumer, K.",
    year: 2018,
    title: "Evolution-guided policy gradient in reinforcement learning",
    venue: "NeurIPS 2018",
    category: "Hybrid Evo-RL",
    relevance: "ERL. Evolutionary population alongside RL actor. RL injects into population, evolution provides diverse rollouts. Most relevant hybrid framing.",
    url: "https://arxiv.org/abs/1805.07917",
  },
];

const CATEGORIES = [...new Set(PAPERS.map((p) => p.category))];

const CATEGORY_COLORS: Record<string, string> = {
  "Core": "bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400 border-red-200 dark:border-red-800",
  "Intrinsic Motivation": "bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-400 border-purple-200 dark:border-purple-800",
  "Multi-Agent RL": "bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400 border-blue-200 dark:border-blue-800",
  "RL Algorithms": "bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400 border-green-200 dark:border-green-800",
  "Evolutionary": "bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-400 border-amber-200 dark:border-amber-800",
  "Hybrid Evo-RL": "bg-teal-100 dark:bg-teal-900/30 text-teal-700 dark:text-teal-400 border-teal-200 dark:border-teal-800",
};

export default function References() {
  const [filterCategory, setFilterCategory] = useState<string | "all">("all");
  const [searchQuery, setSearchQuery] = useState("");

  const filtered = PAPERS.filter((p) => {
    if (filterCategory !== "all" && p.category !== filterCategory) return false;
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      return p.title.toLowerCase().includes(q) || p.authors.toLowerCase().includes(q) ||
        p.relevance.toLowerCase().includes(q) || p.venue.toLowerCase().includes(q);
    }
    return true;
  });

  return (
    <div className="max-w-5xl mx-auto py-10 px-6">
      <h1 className="text-3xl font-bold mb-2">References</h1>
      <p className="text-gray-600 dark:text-gray-400 mb-6">
        {PAPERS.length} papers across the project's research landscape. Each annotated with
        relevance to the evo-reward replication and extension.
      </p>

      {/* Filters */}
      <div className="flex flex-wrap gap-3 mb-6 items-center">
        <input type="text" value={searchQuery} onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Search papers..."
          className="border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 rounded-lg px-3 py-2 text-sm w-56 focus:outline-none focus:border-blue-400 dark:text-gray-200 placeholder:text-gray-400 dark:placeholder:text-gray-500" />

        <button onClick={() => setFilterCategory("all")}
          className={`px-3 py-1 text-xs rounded-full transition ${filterCategory === "all" ? "bg-gray-800 dark:bg-gray-200 text-white dark:text-gray-900" : "bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-700"}`}>
          All
        </button>
        {CATEGORIES.map((cat) => (
          <button key={cat} onClick={() => setFilterCategory(cat === filterCategory ? "all" : cat)}
            className={`px-3 py-1 text-xs rounded-full transition ${filterCategory === cat
              ? "bg-blue-600 text-white"
              : "bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-700"
            }`}>
            {cat}
          </button>
        ))}
      </div>

      {/* Paper cards */}
      <div className="space-y-4">
        {filtered.map((paper) => {
          const catColor = CATEGORY_COLORS[paper.category] || "bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400";
          return (
            <div key={paper.id} className="rounded-lg border border-gray-200 dark:border-gray-700 p-4 bg-white dark:bg-gray-900 hover:border-gray-300 dark:hover:border-gray-600 transition">
              <div className="flex items-start justify-between gap-4">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <span className={`text-[10px] px-1.5 py-0.5 rounded border font-medium ${catColor}`}>
                      {paper.category}
                    </span>
                    <span className="text-xs text-gray-400 dark:text-gray-500">{paper.year}</span>
                  </div>
                  <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-1">
                    {paper.url ? (
                      <a href={paper.url} target="_blank" rel="noopener noreferrer"
                        className="hover:text-blue-600 dark:hover:text-blue-400 transition">
                        {paper.title} <span className="text-xs text-gray-400">↗</span>
                      </a>
                    ) : paper.title}
                  </h3>
                  <div className="text-sm text-gray-500 dark:text-gray-400 mb-2">
                    {paper.authors} — <em>{paper.venue}</em>
                  </div>
                  <p className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                    {paper.relevance}
                  </p>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {filtered.length === 0 && (
        <div className="text-center py-8 text-gray-400 dark:text-gray-500">No papers match your search</div>
      )}

      <div className="mt-8 text-sm text-gray-400 dark:text-gray-500">
        Full annotated bibliography with extended commentary available in the{" "}
        <a href="./docs" className="text-blue-500 dark:text-blue-400 hover:underline">Documentation</a> section.
      </div>
    </div>
  );
}
