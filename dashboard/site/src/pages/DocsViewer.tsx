import { useState, useEffect } from "react";
import ReactMarkdown from "react-markdown";

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
    fetch(`/docs/${selectedDoc}`)
      .then((res) => { if (!res.ok) throw new Error(`Failed`); return res.text(); })
      .then((text) => { setContent(text); setLoading(false); })
      .catch(() => {
        fetch(`../../docs/${selectedDoc}`)
          .then((res) => { if (!res.ok) throw new Error(`Not found`); return res.text(); })
          .then((text) => { setContent(text); setLoading(false); })
          .catch(() => {
            setContent("");
            setError(`Could not load docs/${selectedDoc}. Create a symlink: cd dashboard/site/public && ln -s ../../../docs docs`);
            setLoading(false);
          });
      });
  }, [selectedDoc]);

  return (
    <div className="max-w-6xl mx-auto py-10 px-6">
      <h1 className="text-3xl font-bold mb-2">Documentation Viewer</h1>
      <p className="text-gray-600 dark:text-gray-400 mb-6">Rendered project documentation. Select a document from the sidebar.</p>

      <div className="grid grid-cols-4 gap-6">
        <div className="col-span-1">
          <div className="bg-gray-50 dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-700 p-3">
            <div className="text-xs font-semibold text-gray-400 dark:text-gray-500 uppercase tracking-wide mb-2">docs/</div>
            {DOCS_FILES.map((doc) => (
              <button key={doc.name} onClick={() => setSelectedDoc(doc.name)}
                className={`w-full text-left px-2 py-1.5 rounded text-sm mb-0.5 transition ${
                  selectedDoc === doc.name
                    ? "bg-blue-100 dark:bg-blue-900/40 text-blue-800 dark:text-blue-300 font-medium"
                    : "text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800"
                }`}>
                {doc.label}
              </button>
            ))}
          </div>
        </div>

        <div className="col-span-3">
          <div className="bg-white dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-700 p-6 min-h-96">
            <div className="text-xs text-gray-400 dark:text-gray-500 mb-4 font-mono">docs/{selectedDoc}</div>
            {loading && <div className="text-gray-400 dark:text-gray-500">Loading...</div>}
            {error && (
              <div className="bg-amber-50 dark:bg-amber-950/30 border border-amber-200 dark:border-amber-800 rounded-lg p-4 text-sm text-amber-800 dark:text-amber-300 whitespace-pre-wrap">{error}</div>
            )}
            {!loading && !error && content && (
              <div className="prose prose-sm dark:prose-invert max-w-none
                prose-headings:text-gray-800 dark:prose-headings:text-gray-200
                prose-p:text-gray-700 dark:prose-p:text-gray-300 prose-p:leading-relaxed
                prose-code:text-sm prose-code:bg-gray-100 dark:prose-code:bg-gray-800 prose-code:px-1 prose-code:py-0.5 prose-code:rounded
                prose-pre:bg-gray-900 dark:prose-pre:bg-gray-950 prose-pre:text-gray-100
                prose-table:text-sm prose-th:bg-gray-50 dark:prose-th:bg-gray-800
                prose-td:px-3 prose-td:py-2
                prose-a:text-blue-600 dark:prose-a:text-blue-400 prose-a:no-underline hover:prose-a:underline
                prose-li:text-gray-700 dark:prose-li:text-gray-300
                prose-strong:text-gray-900 dark:prose-strong:text-gray-100">
                <ReactMarkdown
                  components={{
                    a: ({ href, children, ...props }) => {
                      if (href && href.endsWith(".md") && !href.startsWith("http")) {
                        const docName = href.split("/").pop() || href;
                        return <button onClick={() => setSelectedDoc(docName)} className="text-blue-600 dark:text-blue-400 hover:underline cursor-pointer">{children}</button>;
                      }
                      return <a href={href} {...props}>{children}</a>;
                    },
                  }}>
                  {content}
                </ReactMarkdown>
              </div>
            )}
            {!loading && !error && !content && <div className="text-gray-400 dark:text-gray-500">No content loaded</div>}
          </div>
        </div>
      </div>
    </div>
  );
}
