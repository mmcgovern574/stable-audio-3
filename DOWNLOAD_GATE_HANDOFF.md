# Download Gate — Frontend Handoff

The server-side download limit is **live on `soundsauce-prod`**. This doc is everything the frontend (Lovable) needs to wire it up.

## What's deployed (backend — done)

- **Edge function `download-gate`** — `https://ycpyncjkwgtsxvozqnzz.supabase.co/functions/v1/download-gate`
  Checks the free limit server-side, records the download, returns `ok` / `limit_reached`.
- **`downloads` table** — now indexed and de-duped (re-downloading a loop you already have never costs a second credit).
- **Limits** (configurable, see bottom): 10 free downloads per anonymous visitor, plus a 15-per-30-days backstop per IP so clearing Safari / opening a private window / switching browsers on the same network no longer grants a fresh 10.
- Signed-in users pass straight through (no free-tier wall — their credits are handled by your existing `profiles.credits` logic).

## What the frontend must change

Right now the count lives in `localStorage` (`ss_gen_count`, `ss_sid`, …) and nothing hits the server — that's why it resets. The fix: **call the gate before every download, and let the server decide.** Keep localStorage only as a cosmetic counter, never as the source of truth.

## Drop-in snippet

```js
const SUPABASE_URL  = "https://ycpyncjkwgtsxvozqnzz.supabase.co";
const SUPABASE_ANON = "sb_publishable_ycpxAtgoQRMGxgJdj5hO2A_26nmEuyE"; // publishable key, safe in frontend
const GATE = `${SUPABASE_URL}/functions/v1/download-gate`;

// Stable per-browser id (survives reloads; IP backstop covers the cases it doesn't).
function getAnonId() {
  let id = localStorage.getItem("ss_anon_id");
  if (!id) { id = crypto.randomUUID(); localStorage.setItem("ss_anon_id", id); }
  return id;
}

async function checkDownloadAllowed(loopId, accessToken /* optional */) {
  const res = await fetch(GATE, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "apikey": SUPABASE_ANON,
      ...(accessToken ? { Authorization: `Bearer ${accessToken}` } : {}),
    },
    body: JSON.stringify({ anon_id: getAnonId(), loop_id: loopId }),
  });
  return { status: res.status, body: await res.json() };
}

// Wire this to your existing download button.
async function handleDownload(sample) {
  // If you use supabase-js auth:
  // const { data: { session } } = await supabase.auth.getSession();
  const session = null; // <- replace with the real session when signed in
  const { status, body } = await checkDownloadAllowed(sample.id, session?.access_token);

  if (status === 402 || body.error === "limit_reached") {
    openSignupModal();            // <-- your existing signup button / modal
    return;
  }
  if (!body.ok) { console.error("download gate error", body); return; }

  // Allowed → serve the file (the public catalog URL you already have).
  const a = document.createElement("a");
  a.href = sample.audio_path;     // e.g. https://…/object/public/catalog/<id>.mp3
  a.download = `${sample.title || sample.id}.mp3`;
  document.body.appendChild(a); a.click(); a.remove();

  if (typeof body.remaining === "number") updateCreditsUI(body.remaining); // optional
}
```

**Key points**

- Use a **stable per-loop id** for `loop_id` — `sample.id` (the `samples.id` UUID) is the natural choice. Be consistent so re-downloads dedupe correctly.
- Pass the Supabase **session access token** when the user is signed in so they skip the wall.
- The response gives you `used`, `limit`, `remaining` — use those to render "N of 10 free downloads used," and on the last one show the soft "create a free account to keep going" nudge.

## Prompt you can paste into Lovable

> Route the sample download button through our Supabase edge function `download-gate` before serving the file. Add a stable `ss_anon_id` (crypto.randomUUID) in localStorage. On download click, POST `{ anon_id, loop_id: sample.id }` to `https://ycpyncjkwgtsxvozqnzz.supabase.co/functions/v1/download-gate` with the `apikey` header and, if the user is signed in, an `Authorization: Bearer <access_token>` header. If the response status is 402 / `error: "limit_reached"`, open the sign-up modal instead of downloading. Otherwise download the file from `sample.audio_path` and show "{remaining} of 10 free downloads left". Stop using `ss_gen_count` as the enforcement counter — the server response is the source of truth.

## Two things to do / decide

1. **Set the IP salt secret** (2 min, recommended). The function hashes IPs with a salt that currently falls back to a default. In Supabase → Project → Edge Functions → **Secrets**, add `DOWNLOAD_IP_SALT` = any long random string. (Optional: `FREE_DOWNLOAD_LIMIT`, `IP_DOWNLOAD_LIMIT` to change the numbers.)

2. **Phase 2 — private bucket (optional, later).** Your `catalog` and `user-audio` buckets are **public**, so a technical user can still grab the raw `.mp3` URL and bypass the button entirely. Closing that means making the bucket private and serving both playback and downloads via signed URLs — a bigger change that touches your audio player, so it's worth doing deliberately as its own step rather than now. For a free funnel, the button-level gate is usually enough; flag it when you're ready and I'll do the private-bucket migration.
