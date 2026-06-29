# Sound Sauce — Product & Monetization Strategy

Turning the cKz generator + rater into a business. Written 2026-06-27.
TL;DR: the license is green, you've already built ~60% of the MVP, the
one-of-one exclusivity idea is your real moat, and the fastest money does
**not** require the website.

---

## 1. License verdict — you are clear to monetize (the unblock)

Two separate licenses are in play. Don't conflate them:

- **The code** (`stable-audio-3` repo) — **MIT** (`LICENSE` file, © Stability AI 2026).
  Use, modify, sell freely. No constraints worth worrying about.
- **The weights** you load via `from_pretrained("medium-base")` — **Stable Audio 3.0
  Medium**, released 2026-05-20 under the **Stability AI Community License**. This is
  the one that governs your business.

What the Community License actually lets you do:

- **Commercial use is allowed, free, while you (and affiliates, in aggregate) make
  under $1,000,000/yr in revenue.** Above that, the license terminates and you need an
  Enterprise License. You are nowhere near this — it's a champagne problem.
- **"Commercial Purpose" explicitly includes "a hosted service or API."** So your paid
  website, on-demand generation, and even the multi-producer platform are all permitted
  uses — not gray area. This was the thing most likely to bite you, and it doesn't.
- **You own your outputs** and can sell/distribute them however you want. Your "sell
  exclusive samples" model is fine.
- **Fine-tuning is explicitly allowed** — your LoRA is a permitted Derivative Work.
- **Training data is fully licensed.** This is a real, marketable moat: you can advertise
  "100% copyright-clean, lawsuit-proof samples." Suno and Udio cannot — they're in active
  litigation. For a producer deciding whether to put an AI loop in a track they'll
  release commercially, "the training data is licensed" is a genuine buying reason.

Obligations (all minor, do them once):

1. Ship a `NOTICE` file: *"This Stability AI Model is licensed under the Stability AI
   Community License, Copyright © Stability AI Ltd. All Rights Reserved."*
2. **Display "Powered by Stability AI"** prominently on the site (footer / about page).
3. Pass a copy of the license to anyone you distribute the model/derivative to (matters
   for the platform tier).
4. State that you modified the model and how (your LoRA) in the NOTICE.
5. Follow the Acceptable Use Policy; hosted services must have safety filters.

**Watch-item for the platform tier:** the $1M threshold is *aggregate across affiliates*.
If Sound Sauce becomes a platform where many producers earn through your hosted service,
that aggregate can grow faster than you'd expect. Build the Stability Enterprise
relationship *before* you scale the platform — Enterprise also adds legal indemnification,
which is worth a lot once you're selling at volume. Not a blocker now; a calendar note.

> One caveat: I'm not a lawyer. The above is the plain reading of the public license.
> Before the platform tier (3rd-party producers + revenue share), get 30 minutes with
> an IP lawyer and/or email Stability. For tiers 1–2 below you're fine to start today.

---

## 2. Reframe: you have THREE businesses, not one

You've bundled three things with very different build-cost and risk. Separating them is
the whole point of "what do I prioritize."

| # | Business | What it is | Build cost | Validates |
|---|----------|-----------|-----------|-----------|
| A | **Sample seller** | Sell your own curated cKz outputs as packs + one-offs | ~zero | Will anyone pay at all? |
| B | **Sound Sauce app** | Player site: free pool + paid one-of-one exclusives + (later) custom gen | medium | Does the exclusivity model work? |
| C | **Sound Sauce platform** | Train/host *other* producers' models, take a cut | high | Two-sided marketplace |

Do them **in order A → B → C.** Each de-risks the next. Most founders try to build B or C
first and run out of money before learning whether A was ever true.

---

## 3. The real wedge: one-of-one exclusivity

Your best idea is buried in the middle of the brief: *"downloaded once — only that person
has it."* That's not a feature, it's the entire positioning.

- Splice and every sample subscription are **non-exclusive**: 50,000 producers get the
  exact same loop. The loop is disposable; nobody builds a song around something they
  know is in 10,000 other songs.
- The beat world already prices exclusivity: a **non-exclusive lease** sells for ~$20–50,
  the **exclusive rights** to the same beat sell for hundreds to thousands. Producers
  understand and pay for "only I have this."
- Nobody is selling **one-of-one *loops/samples*** because for a human producer, making a
  unique loop per customer doesn't scale. **Your generator makes that marginal cost ~zero.**
  That's the arbitrage. This is the thing you have that Splice structurally cannot copy.

This maps cleanly onto your free/paid split:

- **Free public pool = non-exclusive commons.** A few hundred good cKz loops anyone can
  grab (with a download cap). This is the loss-leader / funnel / SEO surface, not the
  product. It's how people hear the Sound Sauce timbre.
- **Paid = one-of-one exclusives.** Generated, claimed once, then *removed from the pool
  forever*. Provenance ("claimed by you, 2026-07-02, never re-issued") is the product.
  This is your stated "pay for your own novel samples," made concrete.

Decide the promise explicitly and honor it technically: "exclusive" must mean *removed
forever*, enforced in the DB, or the whole differentiator evaporates.

---

## 4. What you already have (more than you think)

- **`app/`** — a FastAPI backend + browse/keep frontend (`static/index.html`) that loads
  aug-8000 once and runs Discover/Refine with a generation queue, density gate, and a
  kept-clip library. **This is the MVP engine skeleton.** It runs locally on your Mac.
- **`sweep_rater/`** — the 3-axis rater. Reframe this: it's **not the product, it's your
  QA line.** Customers never see it. Its job is to guarantee every sample in the catalog
  is a keeper. The browse/keep *UX* is reusable for the consumer feed; the star-rating
  *guts* stay internal. (A stripped "keep/discard swipe" is a nice free-tier engagement
  mechanic, but that's polish.)
- **Supabase is already connected** to this workspace — that's your entire backend for
  auth, the catalog DB, the claim-once logic, and sample storage. You don't need to stand
  up infrastructure from scratch.

So the gap from here to a live product is **not** ML work. It's: a hosted catalog, auth,
Stripe, and the claim-once lock. Standard web-app plumbing.

---

## 5. The latency trap, and how to dodge it

Your model is ~1 min/clip, single-worker, on your Mac. That is fine — **because the free
tier and the exclusives don't need live generation.** Pre-generate the catalog offline
(batch overnight on the M4 Max), curate with the rater, upload static WAVs to Supabase
storage / a CDN. The website then serves *files*, with **no ML in the hot path.** This is
exactly your "pre-generate a ton of samples, operate like a player" instinct — it's the
right call, and it sidesteps GPU hosting entirely for v1.

Only the **custom-generation tier** (upload init audio, pick key/BPM) needs a live GPU,
and that's a v2 you can defer until people are paying. When you get there, rent serverless
GPU (Modal / Replicate / RunPod / fal) rather than running a box 24/7.

---

## 6. Prioritized roadmap

### Step A — Sell now, no website (this week)
Fastest path to first dollar and to the only data that matters: will people pay, and will
they pay *more* for exclusivity?

- Batch-generate a few hundred clips with `app/` overnight; curate the top ~40 with the rater.
- Package: **"Crushed Keyz Vol. 1 — Sound Sauce"** non-exclusive pack (e.g. 20 loops) on
  **Gumroad or BeatStars**, plus **5 one-of-one exclusive loops** at a premium price.
- Drop 3–5 free loops on TikTok/IG/YouTube as "type beat" / "free melody loop" content to
  start the Crushed Keyz brand and collect emails.
- **What you learn:** willingness to pay, the exclusivity premium (do the $X one-offs
  outsell the pack?), and which sounds land — all before writing a line of website code.

### Step B — Sound Sauce website MVP (2–4 weeks)
Only after A shows demand. Player over a **pre-generated** catalog:

- **Free tier:** browse + play the non-exclusive pool, N free downloads (credit counter).
- **Paid tier (Stripe):** claim one-of-one exclusives, higher/again-refilling download cap.
- **Stack:** Next.js on Vercel + Supabase (auth, Postgres catalog, storage) + Stripe.
  Claim-once = a `samples` row with `claimed_by` / `claimed_at`; on purchase, mark claimed
  and drop it from the public query. Reuse `app/static/index.html` for the feed UX.
- **"Powered by Stability AI"** in the footer (license requirement).

### Step C — Custom generation tier (later, after B earns)
Top paid tier: upload init audio, choose key/BPM, generate on demand. This is where the
live GPU backend comes in. Your `/discover` and `/refine` endpoints already define the API.

### Step D — Multi-producer platform (only after the single-producer model is proven)
Train and host other producers' LoRAs, revenue-share or charge for placement. Two hard
problems: it's a two-sided marketplace (need buyers *and* producers), and the license
aggregate-revenue + distribution rules now apply. **Host the weights yourself; never hand
producers their model files** — cleaner for both your moat and the license. Get the lawyer
+ Stability Enterprise conversation done before this goes live.

---

## 7. Risks & open questions

- **Throughput is human-bottlenecked.** Keepers are judged by ear (the auto-scorer was
  dropped for good reasons). Stocking thousands of one-of-ones = real listening hours, or
  accepting the lighter density-gate filter. This is the true operational constraint on
  the "ton of samples" vision — budget for it.
- **Catalog cold-start.** A player with 30 samples feels empty. Pre-generation volume is
  a launch gate; plan a few thousand curated clips before going public with B.
- **Differentiation durability.** Anyone can fine-tune SA3. Your moat is *brand + curation
  + the specific cKz sound + the exclusivity mechanic + licensed-clean positioning* — not
  the model. Lean into all five; none alone is enough.
- **Demand is unproven.** Step A exists to answer this for ~$0. Don't skip it.
- **Pricing the exclusive.** Unknown until tested. A is your price-discovery lab.

---

## 8. The one-line answer to "what do I prioritize first"

**Generate + curate a batch this week and sell it — a non-exclusive pack plus a handful of
one-of-one exclusives — on Gumroad/BeatStars, while dropping free loops to build the brand.
Prove people pay (and pay extra for exclusivity) before you build the site.** The website is
Step B, and you've already built most of its engine. The platform is Step D, and it's the
one place to get a lawyer first.
