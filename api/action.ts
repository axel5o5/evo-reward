// Vercel serverless function: PIN-gates dashboard actions, forwards to
// a GitHub Actions workflow_dispatch. No GCP credentials live here — the
// workflow has them via WIF.
//
// Required env vars (Vercel → Project Settings → Environment Variables):
//   DASHBOARD_PIN        shared secret the user types into the UI
//   GH_DISPATCH_TOKEN    GitHub fine-grained PAT with "Actions: Read+Write"
//                        on this repo only
//   GH_OWNER             e.g. "axel5o5"
//   GH_REPO              e.g. "evo-reward"
//   GH_REF               branch to run the workflow on (default "main")
//
// Returns 202 Accepted on success — the workflow runs async. The dashboard
// monitor page polls gcp-status.json every 30s so VM state changes become
// visible within ~1 min of a successful action.

// Vercel Functions use a Node-native request/response API.
// Use generic types so we don't force a dep on @vercel/node.
type Req = {
  method?: string;
  body: unknown;
};
type Res = {
  status: (code: number) => Res;
  json: (body: unknown) => void;
  end: () => void;
  setHeader: (name: string, value: string) => void;
};

const ALLOWED_ACTIONS = ["stop", "start", "restart", "delete"] as const;
type Action = typeof ALLOWED_ACTIONS[number];

function isAction(x: unknown): x is Action {
  return typeof x === "string" && (ALLOWED_ACTIONS as readonly string[]).includes(x);
}

// Timing-safe equality so a leaked server can't be brute-forced via
// response-time side channels. 6-12 chars of entropy is usually enough
// for a friends-of-family PIN but don't rely on <8 for anything real.
function safeEq(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}

export default async function handler(req: Req, res: Res): Promise<void> {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    res.status(405).json({ error: "method_not_allowed" });
    return;
  }

  const { pin, action } = (req.body ?? {}) as { pin?: unknown; action?: unknown };
  const expectedPin = process.env.DASHBOARD_PIN ?? "";

  if (!expectedPin) {
    res.status(500).json({ error: "server_misconfigured", detail: "DASHBOARD_PIN not set" });
    return;
  }
  if (typeof pin !== "string" || !safeEq(pin, expectedPin)) {
    res.status(401).json({ error: "invalid_pin" });
    return;
  }
  if (!isAction(action)) {
    res.status(400).json({ error: "invalid_action", allowed: ALLOWED_ACTIONS });
    return;
  }

  const token = process.env.GH_DISPATCH_TOKEN;
  const owner = process.env.GH_OWNER;
  const repo = process.env.GH_REPO;
  const ref = process.env.GH_REF ?? "main";
  if (!token || !owner || !repo) {
    res.status(500).json({ error: "server_misconfigured",
      detail: "GH_DISPATCH_TOKEN, GH_OWNER, or GH_REPO missing" });
    return;
  }

  // Short tag so the user can cross-reference this request with the
  // workflow run in the Actions tab.
  const requestId = Math.random().toString(36).slice(2, 8);

  const ghUrl = `https://api.github.com/repos/${owner}/${repo}/actions/workflows/gcp-action.yml/dispatches`;
  const ghRes = await fetch(ghUrl, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
      Accept: "application/vnd.github+json",
      "X-GitHub-Api-Version": "2022-11-28",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      ref,
      inputs: { action, request_id: requestId },
    }),
  });

  if (!ghRes.ok) {
    const detail = await ghRes.text();
    res.status(502).json({ error: "github_dispatch_failed", status: ghRes.status, detail });
    return;
  }

  res.status(202).json({
    ok: true,
    action,
    request_id: requestId,
    hint: "Workflow dispatched. State will update on the dashboard within ~1 minute.",
  });
}
