#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Instagram reel transcriber — single or batch mode
#
#   Single:
#     bash scripts/transcribe.sh https://www.instagram.com/reel/ABC123/
#
#   Batch (multiple URLs):
#     bash scripts/transcribe.sh URL1 URL2 URL3
#
#   Batch (from file, one URL per line):
#     bash scripts/transcribe.sh -f urls.txt
#
#   What it does:
#     1. Starts the uvicorn server in the background (if not running)
#     2. Submits all Instagram reel URLs in parallel
#     3. Polls all jobs until finished
#     4. Downloads SRT + TXT files to data/output/
#     5. Prints a summary + all transcripts
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Config ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BASE_URL="http://localhost:8000"
DATA_DIR="$PROJECT_DIR/data/output"
SERVER_LOG="$PROJECT_DIR/data/server.log"
VENV_PYTHON="$PROJECT_DIR/.venv/Scripts/python.exe"
HEALTH_URL="$BASE_URL/health"
JOB_POLL_INTERVAL=3

# ── Parse args ──────────────────────────────────────────────────────────────
URLS=()
TRANSLATE=false
LANG=""

# Extract --translate and --lang flags
while [ $# -gt 0 ]; do
  case "$1" in
    --translate)
      TRANSLATE=true
      shift
      ;;
    --lang)
      LANG="$2"
      shift 2
      ;;
    --lang=*)
      LANG="${1#--lang=}"
      shift
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done
set -- "${ARGS[@]}"

if [ $# -ge 2 ] && [ "$1" = "-f" ]; then
  # Read URLs from file
  FILE="$2"
  if [ ! -f "$FILE" ]; then
    echo "✗ File not found: $FILE"
    exit 1
  fi
  while IFS= read -r line; do
    line=$(echo "$line" | xargs)  # trim whitespace
    [ -z "$line" ] && continue
    [[ "$line" == \#* ]] && continue  # skip comments
    URLS+=("$line")
  done < "$FILE"
elif [ $# -ge 1 ]; then
  # URLs as arguments
  URLS=("$@")
else
  echo "Usage:"
  echo "  $0 [--lang CODE] [--translate] <url1> [url2] [url3] ..."
  echo "  $0 [--lang CODE] [--translate] -f <file_with_urls.txt>"
  echo ""
  echo "  --lang CODE   ISO-639-1 language code (en, ar, fr, es, etc.)"
  echo "  --translate   Translate any language to English (Groq translations endpoint)"
  echo "  File format:  one Instagram URL per line (# comments and blank lines ignored)"
  echo ""
  echo "Examples:"
  echo "  $0 https://www.instagram.com/reel/ABC123/"
  echo "  $0 --lang ar https://www.instagram.com/reel/ABC123/"
  echo "  $0 --lang en --translate https://www.instagram.com/reel/ABC123/"
  echo "  $0 -f my_reels.txt"
  echo "  $0 --lang ar -f my_reels.txt"
  exit 1
fi

TOTAL=${#URLS[@]}
if [ "$TOTAL" -eq 0 ]; then
  echo "✗ No URLs provided"
  exit 1
fi

# ── Ensure venv & deps ──────────────────────────────────────────────────────
if [ ! -d "$PROJECT_DIR/.venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv "$PROJECT_DIR/.venv"
  "$PROJECT_DIR/.venv/Scripts/pip" install -r "$PROJECT_DIR/requirements.txt"
fi

# ── Start server if not already running ─────────────────────────────────────
if curl -sf "$HEALTH_URL" > /dev/null 2>&1; then
  echo "✓ Server already running at $BASE_URL"
else
  echo "Starting server in background..."
  cd "$PROJECT_DIR"
  "$VENV_PYTHON" -m uvicorn app.main:app \
    --host 0.0.0.0 --port 8000 \
    > "$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  echo "$SERVER_PID" > "$PROJECT_DIR/data/server.pid"

  echo -n "Waiting for server"
  for i in $(seq 1 30); do
    if curl -sf "$HEALTH_URL" > /dev/null 2>&1; then
      echo " ✓ (ready)"
      break
    fi
    echo -n "."
    sleep 1
  done

  if ! curl -sf "$HEALTH_URL" > /dev/null 2>&1; then
    echo " ✗ Server failed to start. Check $SERVER_LOG"
    exit 1
  fi
fi

# ── Ensure output dir ──────────────────────────────────────────────────────
mkdir -p "$DATA_DIR"

# ── Submit all URLs ────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Submitting $TOTAL reel(s)..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

declare -A JOB_IDS   # job_id → url
declare -A JOB_STATUS # job_id → status

IDX=0
for URL in "${URLS[@]}"; do
  IDX=$((IDX + 1))
  echo -n "  [$IDX/$TOTAL] $URL → "

  # Build JSON body with optional lang and translate fields
  BODY="{\"url\": \"$URL\"}"
  if [ -n "$LANG" ]; then
    BODY=$(echo "$BODY" | sed 's/}$/, "lang": "'"$LANG"'"}/')
  fi
  if [ "$TRANSLATE" = true ]; then
    BODY=$(echo "$BODY" | sed 's/}$/, "translate": true}/')
  fi

  RESPONSE=$(curl -sf -X POST "$BASE_URL/transcribe" \
    -H "Content-Type: application/json" \
    -d "$BODY" 2>/dev/null) || {
    echo "✗ submit failed"
    continue
  }

  JOB_ID=$(echo "$RESPONSE" | grep -o '"job_id":"[^"]*"' | cut -d'"' -f4)
  if [ -z "$JOB_ID" ]; then
    echo "✗ no job_id in response"
    continue
  fi

  JOB_IDS[$JOB_ID]="$URL"
  JOB_STATUS[$JOB_ID]="queued"
  echo "✓ $JOB_ID"
done

SUBMITTED=${#JOB_IDS[@]}
if [ "$SUBMITTED" -eq 0 ]; then
  echo ""
  echo "✗ No jobs were submitted successfully"
  exit 1
fi

echo ""
if [ "$TRANSLATE" = true ]; then
  echo "  Mode: TRANSLATE → English"
elif [ -n "$LANG" ]; then
  echo "  Language: $LANG"
fi
echo "  $SUBMITTED/$TOTAL jobs submitted. Polling..."

# ── Poll all jobs until all done ────────────────────────────────────────────
DONE_COUNT=0
FAIL_COUNT=0

while [ "$DONE_COUNT" -lt "$SUBMITTED" ]; do
  sleep "$JOB_POLL_INTERVAL"

  for JOB_ID in "${!JOB_IDS[@]}"; do
    CURRENT="${JOB_STATUS[$JOB_ID]}"
    [ "$CURRENT" = "done" ] || [ "$CURRENT" = "failed" ] && continue

    STATUS=$(curl -sf "$BASE_URL/jobs/$JOB_ID" 2>/dev/null \
      | grep -o '"status":"[^"]*"' | cut -d'"' -f4) || continue

    case "$STATUS" in
      done)
        JOB_STATUS[$JOB_ID]="done"
        DONE_COUNT=$((DONE_COUNT + 1))
        URL_SHORT="${JOB_IDS[$JOB_ID]}"
        echo "  ✓ [$DONE_COUNT/$SUBMITTED] Done: $URL_SHORT"
        ;;
      failed)
        JOB_STATUS[$JOB_ID]="failed"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        DONE_COUNT=$((DONE_COUNT + 1))
        URL_SHORT="${JOB_IDS[$JOB_ID]}"
        echo "  ✗ [$DONE_COUNT/$SUBMITTED] Failed: $URL_SHORT"
        ;;
    esac
  done
done

# ── Fetch all results ──────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  FETCHING RESULTS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

SUCCESS_COUNT=0
FAIL_FETCH=0

for JOB_ID in "${!JOB_IDS[@]}"; do
  URL="${JOB_IDS[$JOB_ID]}"
  STATUS="${JOB_STATUS[$JOB_ID]}"

  if [ "$STATUS" = "failed" ]; then
    echo ""
    echo "  ✗ SKIPPED (job failed): $URL"
    FAIL_FETCH=$((FAIL_FETCH + 1))
    continue
  fi

  SAFE_ID=$(echo "$JOB_ID" | sed 's/[^a-zA-Z0-9_-]/_/g')
  SRT_FILE="$DATA_DIR/transcript_${SAFE_ID}.srt"
  TXT_FILE="$DATA_DIR/transcript_${SAFE_ID}.txt"

  curl -sf "$BASE_URL/jobs/$JOB_ID/srt" -o "$SRT_FILE" 2>/dev/null || {
    echo "  ✗ Failed to download SRT for $JOB_ID"
    FAIL_FETCH=$((FAIL_FETCH + 1))
    continue
  }
  curl -sf "$BASE_URL/jobs/$JOB_ID/txt" -o "$TXT_FILE" 2>/dev/null || {
    echo "  ✗ Failed to download TXT for $JOB_ID"
    FAIL_FETCH=$((FAIL_FETCH + 1))
    continue
  }

  SUCCESS_COUNT=$((SUCCESS_COUNT + 1))

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  [$SUCCESS_COUNT] $URL"
  echo "  SRT: $SRT_FILE"
  echo "  TXT: $TXT_FILE"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo ""
  echo "── Transcript ──"
  cat "$TXT_FILE"
  echo ""
  echo ""
  echo "── Subtitles ──"
  cat "$SRT_FILE"
done

# ── Final summary ──────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  BATCH COMPLETE 🎉"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Total:    $TOTAL"
echo "  Success:  $SUCCESS_COUNT"
echo "  Failed:   $((TOTAL - SUCCESS_COUNT))"
echo "  Output:   $DATA_DIR/"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
