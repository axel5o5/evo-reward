// Vercel serverless function: reads gcp-status.json from the private
// gcp-status branch using a server-side GitHub PAT, and returns it to
// the browser. Avoids exposing the token to the client.
//
// Uses the Contents API (api.github.com) rather than raw.githubusercontent.com.
// raw.* is served via Fastly with ~5 min per-POP caching, which caused the
// dashboard to silently show stale data for up to 5 min after a monitor
// publish — Vercel's iad1 region hit a different POP than the user's
// browser, so freshness diverged by vantage point. The Contents API
// isn't Fastly-cached; served direct from api.github.com.
//
// Required env vars (shared with /api/action):
//   GH_DISPATCH_TOKEN   fine-grained PAT with Contents: Read on this repo
//   GH_OWNER            GitHub username/org that owns the repo
//   GH_REPO             repo name, e.g. "evo-reward"

type Req = { method?: string };
type Res = {
  status: (code: number) => Res;
  json: (body: unknown) => void;
  setHeader: (name: string, value: string) => void;
  send: (body: string) => void;
};

const STATUS_BRANCH = "gcp-status";
const STATUS_FILE = "gcp-status.json";

export default async function handler(req: Req, res: Res): Promise<void> {
  if (req.method !== "GET") {
    res.setHeader("Allow", "GET");
    res.status(405).json({ error: "method_not_allowed" });
    return;
  }

  const token = process.env.GH_DISPATCH_TOKEN;
  const owner = process.env.GH_OWNER;
  const repo = process.env.GH_REPO;
  if (!token || !owner || !repo) {
    res.status(500).json({
      error: "server_misconfigured",
      detail: "GH_DISPATCH_TOKEN, GH_OWNER, or GH_REPO missing",
    });
    return;
  }

  const url = `https://api.github.com/repos/${owner}/${repo}/contents/${STATUS_FILE}?ref=${STATUS_BRANCH}`;
  const r = await fetch(url, {
    headers: {
      Authorization: `Bearer ${token}`,
      // The "raw" media type returns the file body directly rather than
      // the usual metadata+base64 envelope. Smaller payload, no decode.
      Accept: "application/vnd.github.raw+json",
      "X-GitHub-Api-Version": "2022-11-28",
    },
    // Belt-and-suspenders: instruct undici not to reuse any cached
    // response, even though Node fetch doesn't cache by default.
    cache: "no-store",
  });

  if (!r.ok) {
    const detail = r.status === 404
      ? `${STATUS_BRANCH} branch or ${STATUS_FILE} missing. Run the monitor workflow once.`
      : await r.text();
    // 404 pass-through so the UI can show a meaningful "not initialized" state.
    res.status(r.status === 404 ? 404 : 502).json({
      error: "fetch_failed",
      upstream_status: r.status,
      detail,
    });
    return;
  }

  const body = await r.text();
  res.setHeader("Content-Type", "application/json");
  // Short edge cache so N dashboard clients share a single GitHub fetch
  // per 10s window; reduces rate-limit pressure on the PAT.
  // Kept tight since monitor publishes can land any time.
  res.setHeader("Cache-Control", "public, s-maxage=10, stale-while-revalidate=20");
  res.send(body);
}
