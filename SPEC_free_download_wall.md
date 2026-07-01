# SoundSauce — Free Download Wall Spec

**Goal:** Let any visitor download up to **10 loops** with no account. On the 11th attempt, block the download and require sign-up. Enforcement is **server-side and hardened** — clearing cookies should not cheaply reset the counter.

**Scope:** Anonymous browsing → free download counting → the sign-up gate. Stops at account creation. (Stripe / paid tier and cap refills are deliberately out of scope here.)

---

## 1. Stack assumptions

Per the SoundSauce strategy doc: **Next.js on Vercel + Supabase (Auth, Postgres, Storage)**. No new infrastructure introduced.

The one rule that makes the whole thing work: **audio files are never served from a public URL.** Every download flows through a server endpoint that checks the counter first, then hands back a short-lived signed URL. If the file is publicly linkable, the counter is decorative.

---

## 2. Identity: how we recognize a visitor without an account

Use **Supabase Anonymous Auth** (`supabase.auth.signInAnonymously()`) on first page load. This gives every visitor a real `auth.users` row and a JWT — no email, no friction — and crucially it means the eventual sign-up is an *upgrade of the same user ID*, so download history carries over automatically (see §7).

Anonymous auth alone isn't enough to harden the limit (a user can wipe storage and get a fresh anon ID). So the counter is enforced against **two keys**, and a download counts against the *higher* of the two:

- **`user_id`** — the Supabase anon (or real) user. Primary key for the UX counter.
- **`ip_hash`** — a salted hash of the request IP, optionally combined with a coarse client fingerprint (User-Agent + a few stable headers). Secondary backstop so a cookie wipe doesn't grant a fresh 10.

The IP backstop is intentionally a little loose (shared IPs exist — offices, NAT, mobile carriers). Set its ceiling higher than the per-user limit so legitimate shared networks aren't punished. Recommended: **per-user = 10, per-IP = 30 per rolling 30 days.**

---

## 3. Data model

```sql
-- One row per download event. The source of truth.
create table public.downloads (
  id           uuid primary key default gen_random_uuid(),
  user_id      uuid not null references auth.users(id) on delete cascade,
  loop_id      text not null,                 -- which catalog clip
  ip_hash      text not null,                 -- salted hash of request IP
  fingerprint  text,                          -- optional coarse client hash
  created_at   timestamptz not null default now()
);

create index downloads_user_idx on public.downloads (user_id, created_at desc);
create index downloads_ip_idx   on public.downloads (ip_hash, created_at desc);

-- Lock the table down: clients can never write or read it directly.
alter table public.downloads enable row level security;
-- (No policies = no client access. Only the service-role server code touches it.)
```

Counts are derived, not stored, so they can't be tampered with:

```sql
-- per-user lifetime count
select count(*) from public.downloads where user_id = $1;

-- per-IP rolling-window count
select count(*) from public.downloads
where ip_hash = $1 and created_at > now() - interval '30 days';
```

If `count(*)` over the full table ever gets slow, swap to a maintained `download_counters` summary table updated in the same transaction. Not needed at launch volumes.

**Config (env vars):**

```
FREE_DOWNLOAD_LIMIT=10       # per user
IP_DOWNLOAD_LIMIT=30         # per ip_hash / 30 days
IP_HASH_SALT=<secret>        # so ip_hash isn't reversible
SIGNED_URL_TTL_SECONDS=120   # download link lifetime
```

---

## 4. Download flow

A single server endpoint owns the decision. Pseudocode for `POST /api/download` (Next.js route handler or Supabase Edge Function, running with the **service role** key):

```ts
// 1. Identify the caller from their JWT (anon or real user).
const userId = getUserIdFromJWT(req);            // 401 if no valid session
const ipHash = sha256(clientIp(req) + IP_HASH_SALT);
const { loopId } = await req.json();

// 2. Derive both counts.
const userCount = await countDownloads({ userId });
const ipCount   = await countDownloads({ ipHash, sinceDays: 30 });

const isAnonymous = isAnonUser(userId);          // Supabase flags this on the user

// 3. Gate — only anonymous users are capped. Signed-up users pass (this spec
//    stops at the wall; the paid tier decides what happens past it).
if (isAnonymous && (userCount >= FREE_DOWNLOAD_LIMIT || ipCount >= IP_DOWNLOAD_LIMIT)) {
  return json(402, {
    error: "limit_reached",
    reason: userCount >= FREE_DOWNLOAD_LIMIT ? "user" : "ip",
    used: userCount,
    limit: FREE_DOWNLOAD_LIMIT,
  });
}

// 4. Record the download, then issue a short-lived signed URL.
//    Record FIRST so a crash can only over-count, never give a free extra.
await insertDownload({ userId, loopId, ipHash, fingerprint });
const url = await supabase.storage
  .from("loops")
  .createSignedUrl(`${loopId}.wav`, SIGNED_URL_TTL_SECONDS);

return json(200, { url, used: userCount + 1, limit: FREE_DOWNLOAD_LIMIT });
```

**Why record-before-serve:** the failure mode is then "user got blocked one download early," not "user got unlimited free downloads." Always fail toward the cap.

**Idempotency note:** if you want re-downloading the *same* loop to not burn a second credit, check for an existing `(user_id, loop_id)` row before inserting and skip the increment when found. Decide this product-side; the strategy's "downloaded once" exclusivity model probably wants each *distinct* loop to count once, so this is recommended.

---

## 5. The wall (client UX)

The client keeps a cheap local mirror of `used/limit` (returned by every `/api/download` response) purely to render the UI — it is **never** the thing that enforces anything.

- **Downloads 1–9:** button reads normally; a small counter shows e.g. *"3 of 10 free downloads used."*
- **Download 10 (last one):** after it completes, show a soft heads-up: *"That was your last free download — create a free account to keep going."*
- **Download 11 (blocked):** the `402 limit_reached` response swaps the download button for the gate:

  > **You've used all 10 free downloads.**
  > Create a free account to keep downloading.
  > **[ Sign up free ]** · *Already have an account? Log in*

Because the limit is server-enforced, a user who edits localStorage to fake a lower count just gets a `402` when they actually click — the UI lie buys them nothing.

---

## 6. Anti-abuse — what each layer stops

| Bypass attempt | Defense |
|---|---|
| Clear cookies / localStorage → new anon user | `ip_hash` backstop still counts them |
| Hotlink / share the raw file URL | No public URLs; signed URLs expire in ~2 min |
| Script the download endpoint | Per-IP rolling limit + standard rate limiting on the route |
| Spoof a lower `used` count client-side | Count is recomputed server-side every call; client number is cosmetic |
| VPN / IP rotation to dodge the IP cap | Accept it as residual — this is a *funnel*, not DRM; per-user cap still applies and the goal is conversion, not perfect prevention |

Set expectations accordingly: the wall should be **annoying to bypass, not impossible.** Over-hardening (device fingerprinting, captchas) hurts real users more than it stops the determined few, and the whole point of the free pool is reach.

---

## 7. Conversion: anonymous → signed up

When the gated visitor signs up, **upgrade the existing anonymous user in place** rather than creating a new one:

```ts
// They already have an anon session; attach credentials to the SAME user id.
await supabase.auth.updateUser({ email, password });
// or link an OAuth identity via linkIdentity()
```

The `user_id` doesn't change, so their 10 download rows stay attached and their history is intact. After upgrade, `isAnonymous` flips to false and the free cap no longer gates them — they hand off to whatever the paid/registered tier defines (out of scope here).

If they instead log into a *pre-existing* account, that's a different `user_id`; their anon download history is abandoned (fine — they've already converted).

---

## 8. Edge cases & decisions to confirm

- **Does re-downloading the same loop cost a credit?** Recommended **no** (count distinct loops) — see §4 idempotency. Confirm against the exclusivity model.
- **Per-user limit: lifetime or rolling?** Spec assumes **lifetime** for the free anon cap (you only get 10, ever, without an account). The *IP* backstop is rolling 30 days so shared networks recover.
- **Shared-IP false positives** (schools, offices): the higher per-IP ceiling (30) absorbs most; surface a support path on the gate copy if it bites.
- **Signed-URL TTL:** 120s is enough to start a download, short enough to kill hotlinking. Tune if large files time out on slow connections.

---

## 9. Implementation checklist

1. Enable **Anonymous Sign-ins** in Supabase Auth settings.
2. Run the §3 migration; confirm RLS leaves the table client-inaccessible.
3. Move loop audio into a **private** Storage bucket (`loops`).
4. Build `POST /api/download` with the §4 logic (service-role key, server-only).
5. Add per-IP rate limiting to the route (Vercel middleware or Supabase).
6. Wire the client counter + the gate swap on `402` (§5).
7. Implement the in-place upgrade on sign-up (§7).
8. Test the bypass matrix in §6 manually before launch.
