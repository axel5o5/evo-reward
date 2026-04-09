import { useState, useEffect } from "react";
import ReactMarkdown from "react-markdown";

// Known docs files — we fetch them at runtime since they're in the project root
const DOCS_FILES = [
  { name: "background.md", label: "Background" },
  { name: "technical-spec-kd-replication.md", label: "Technical Spec" },
  { name: "interfaces.md", label: "Interfaces" },
  { name: "emevo-diff.md", label: "emevo Diff" },
  { name: "development-roadmap.md", label: "Development Roadmap" },
  { name: "experimental-plan.md", label: "Experimental Plan" },
  { name: "full-extension-design-doc.md", label: "Extension Design" },
  { name: "dashboard-design.md", label: "Dashboard Design" },
];

export default function DocsViewer() {
  const [selectedDoc, setSelectedDoc] = useState(DOCS_FILES[0].name);
  const [content, setContent] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);

    // In dev mode, docs are at the project root. We try to fetch via a relative path.
    // Since Vite serves from dashboard/site/, we use ../../docs/ to reach project root.
    // In production, docs would need to be copied to public/ during build.
    fetch(`/docs/${selectedDoc}`)
      .then((res) => {
        if (!res.ok) throw new Error(`Failed to load ${selectedDoc}`);
        return res.text();
      })
      .then((text) => {
        setContent(text);
        setLoading(false);
      })
      .catch(() => {
        // Fallback: try alternate paths
        fetch(`../../docs/${selectedDoc}`)
          .then((res) => {
            if (!res.ok) throw new Error(`Not found`);
            return res.text();
          })
          .then((text) => {
            setContent(text);
            setLoading(false);
          })
          .catch(() => {
            setContent("");
            setError(
              `Could not load docs/${selectedDoc}. ` +
              `To enable the docs viewer, either:\n` +
              `1. Create a symlink: cd dashboard/site/public && ln -s ../../../docs docs\n` +
              `2. Or copy docs to dashboard/site/public/docs/`
            );
            setLoading(false);
          });
      });
  }, [selectedDoc]);

  return (
    <div className="max-w-6xl mx-auto py-10 px-6">
      <h1 className="text-3xl font-bold mb-2">Documentation Viewer</h1>
      <p className="text-gray-600 mb-6">
        Rendered project documentation. Select a document from the sidebar.
      </p>

      <div className="grid grid-cols-4 gap-6">
        {/* Sidebar */}
        <div className="col-span-1">
          <div className="bg-gray-50 rounded-lg border border-gray-200 p-3">
            <div className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-2">
              docs/
            </div>
            {DOCS_FILES.map((doc) => (
              <button
                key={doc.name}
                onClick={() => setSelectedDoc(doc.name)}
                className={`w-full text-left px-2 py-1.5 rounded text-sm mb-0.5 transition ${
                  selectedDoc === doc.name
                    ? "bg-blue-100 text-blue-800 font-medium"
                    : "text-gray-600 hover:bg-gray-100"
                }`}
              >
                {doc.label}
              </button>
            ))}
          </div>
        </div>

        {/* Content */}
        <div className="col-span-3">
          <div className="bg-white rounded-lg border border-gray-200 p-6 min-h-96">
            <div className="text-xs text-gray-400 mb-4 font-mono">docs/{selectedDoc}</div>

            {loading && <div className="text-gray-400">Loading...</div>}

            {error && (
              <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 text-sm text-amber-800 whitespace-pre-wrap">
                {error}
              </div>
            )}

            {!loading && !error && content && (
              <div className="prose prose-sm max-w-none
                prose-headings:text-gray-800 prose-h1:text-2xl prose-h2:text-xl prose-h3:text-lg
                prose-p:text-gray-700 prose-p:leading-relaxed
                prose-code:text-sm prose-code:bg-gray-100 prose-code:px-1 prose-code:py-0.5 prose-code:rounded
                prose-pre:bg-gray-900 prose-pre:text-gray-100
                prose-table:text-sm
                prose-th:bg-gray-50 prose-th:px-3 prose-th:py-2
                prose-td:px-3 prose-td:py-2 prose-td:border-gray-200
                prose-a:text-blue-600 prose-a:no-underline hover:prose-a:underline
                prose-li:text-gray-700">
                <ReactMarkdown
                  components={{
                    // Handle cross-links: if an href points to another .md file, navigate within the viewer
                    a: ({ href, children, ...props }) => {
                      if (href && href.endsWith(".md") && !href.startsWith("http")) {
                        const docName = href.split("/").pop() || href;
                        return (
                          <button
                            onClick={() => setSelectedDoc(docName)}
                            className="text-blue-600 hover:underline cursor-pointer"
                          >
                            {children}
                          </button>
                        );
                      }
                      return <a href={href} {...props}>{children}</a>;
                    },
                  }}
                >
                  {content}
                </ReactMarkdown>
              </div>
            )}

            {!loading && !error && !content && (
              <div className="text-gray-400">No content loaded</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
