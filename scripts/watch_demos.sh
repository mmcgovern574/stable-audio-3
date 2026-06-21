#!/usr/bin/env bash
# Mirror the CURRENT training run's demos into a per-run folder, so demo ratings
# are keyed by a unique path and never collide with other runs (demo filenames
# like demo_cfg_4_00000501.wav repeat across every run).
#
# Usage:
#   bash scripts/watch_demos.sh <run-name>
# Then point the rater at:  sweep_rater/campaigns/<run-name>
#
# Only files modified in the last 4h are copied, so the stale demo_cfg_4_*.wav
# left in the repo root from previous runs are ignored. cp -n copies each demo
# once. Leave running alongside training; Ctrl-C to stop.
set -euo pipefail
cd "$(dirname "$0")/.."

RUN="${1:-demos_$(date +%Y%m%d_%H%M)}"
DEST="sweep_rater/campaigns/$RUN"
mkdir -p "$DEST"

# Marker = 15 min ago, so we copy only demos written from ~now onward (this run),
# never the stale same-named demos left in the repo root by previous runs.
MARKER="$(mktemp)"
TS=$(date -v-15M +%Y%m%d%H%M 2>/dev/null || date -d '15 minutes ago' +%Y%m%d%H%M 2>/dev/null)
[ -n "$TS" ] && touch -t "$TS" "$MARKER" || touch "$MARKER"

echo "Mirroring this run's demos -> $DEST"
echo "Rate with:  python sweep_rater/rate_server.py --folder $DEST --segments data/demo_prompts_ckz_exact.txt --segment-seconds 12"
echo "(Ctrl-C to stop)"

while true; do
  find . -maxdepth 1 -name 'demo_cfg_4_*.wav' -newer "$MARKER" -exec cp -n {} "$DEST/" \; 2>/dev/null || true
  sleep 30
done
