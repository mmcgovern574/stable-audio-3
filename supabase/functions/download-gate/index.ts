// Deployed to soundsauce-prod (project ycpyncjkwgtsxvozqnzz) as edge function `download-gate`.
// Server-side free-download wall: 10 free downloads per anonymous visitor (keyed to
// anon_id + hashed IP), signed-in users pass through. See DOWNLOAD_GATE_HANDOFF.md.
import { createClient } from "jsr:@supabase/supabase-js@2";

// ---- Config (override via edge function secrets) ----
const FREE_LIMIT = Number(Deno.env.get("FREE_DOWNLOAD_LIMIT") ?? "10"); // per anonymous visitor
const IP_LIMIT = Number(Deno.env.get("IP_DOWNLOAD_LIMIT") ?? "15");    // per IP / 30 days (backstop)
const IP_SALT = Deno.env.get("DOWNLOAD_IP_SALT") ?? "soundsauce-change-this-salt";
const WINDOW_MS = 30 * 24 * 60 * 60 * 1000;

const cors: Record<string, string> = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
};

function json(status: number, body: unknown) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { ...cors, "Content-Type": "application/json" },
  });
}

async function sha256(s: string): Promise<string> {
  const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(s));
  return [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") return new Response("ok", { headers: cors });
  if (req.method !== "POST") return json(405, { error: "method_not_allowed" });

  try {
    const admin = createClient(
      Deno.env.get("SUPABASE_URL")!,
      Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!,
    );

    const body = await req.json().catch(() => ({}));
    const loop_id: string | undefined = body.loop_id;
    const anon_id: string | undefined = body.anon_id;
    if (!loop_id) return json(400, { error: "missing_loop_id" });

    // Is this a real signed-in (non-anonymous) user?
    let userId: string | null = null;
    let isAnon = true;
    const authz = req.headers.get("Authorization");
    if (authz?.startsWith("Bearer ")) {
      const token = authz.slice(7);
      const { data } = await admin.auth.getUser(token);
      if (data?.user && !(data.user as any).is_anonymous) {
        userId = data.user.id;
        isAnon = false;
      }
    }

    const ip = (req.headers.get("x-forwarded-for") ?? "").split(",")[0].trim() || "0.0.0.0";
    const ipHash = await sha256(ip + IP_SALT);

    // ---- Signed-in users: record only, no free-tier wall here ----
    if (!isAnon && userId) {
      const { error } = await admin.from("downloads").insert({ user_id: userId, loop_id, ip_hash: ipHash });
      // 23505 = unique violation = already downloaded this loop; that's fine.
      if (error && error.code !== "23505") return json(500, { error: "insert_failed", detail: error.message });
      return json(200, { ok: true, signedIn: true });
    }

    // ---- Anonymous visitors: enforce the free wall ----
    if (!anon_id) return json(400, { error: "missing_anon_id" });

    // Has this visitor already grabbed THIS loop? Re-downloads are free.
    const { count: dupe } = await admin.from("downloads")
      .select("id", { count: "exact", head: true })
      .eq("anon_id", anon_id).eq("loop_id", loop_id);

    // Distinct loops this visitor has taken (each loop is one row max).
    const { count: anonCount } = await admin.from("downloads")
      .select("id", { count: "exact", head: true })
      .eq("anon_id", anon_id);

    // IP backstop: total downloads from this network in the window.
    const { count: ipCount } = await admin.from("downloads")
      .select("id", { count: "exact", head: true })
      .eq("ip_hash", ipHash)
      .gte("created_at", new Date(Date.now() - WINDOW_MS).toISOString());

    const used = anonCount ?? 0;
    const isDupe = (dupe ?? 0) > 0;

    if (!isDupe && (used >= FREE_LIMIT || (ipCount ?? 0) >= IP_LIMIT)) {
      return json(402, {
        error: "limit_reached",
        reason: used >= FREE_LIMIT ? "user" : "ip",
        used,
        limit: FREE_LIMIT,
      });
    }

    if (!isDupe) {
      const { error } = await admin.from("downloads").insert({ anon_id, loop_id, ip_hash: ipHash });
      if (error && error.code !== "23505") return json(500, { error: "insert_failed", detail: error.message });
    }

    const newUsed = isDupe ? used : used + 1;
    return json(200, {
      ok: true,
      used: newUsed,
      limit: FREE_LIMIT,
      remaining: Math.max(0, FREE_LIMIT - newUsed),
    });
  } catch (e) {
    return json(500, { error: String(e) });
  }
});
