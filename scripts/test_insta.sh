#!/usr/bin/env bash
# Submit an Instagram URL to the backend and poll until it's done.
#
#   ./test_insta.sh https://www.instagram.com/reel/ABC123/
#
# Env:
#   BASE_URL   (default http://localhost:8000)
#   API_KEY    (only if the server was started with API_KEY set)
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
URL="${1:?Usage: $0 <instagram_url> [job_id]}"
JOB_ID="${2:-}"
AUTH=()
if [ -n "${API_KEY:-}" ]; then
  AUTH=(-H "X-API-Key: $API_KEY")
fi

BODY=$(jq -n --arg url "$URL" --arg jobId "$JOB_ID" '{url: $url, job_id: $jobId}')
echo "→ POST $BASE_URL/transcribe"
RESP=$(curl -sf -X POST "$BASE_URL/transcribe" -H "Content-Type: application/json" "${AUTH[@]}" -d "$BODY")
echo "$RESP"

JOB=$(echo "$RESP" | jq -r .job_id)
[ -n "$JOB" ] || { echo "no job_id in response" >&2; exit 1; }

echo "→ polling /jobs/$JOB"
while :; do
  STATE=$(curl -sf "${AUTH[@]}" "$BASE_URL/jobs/$JOB" | jq -r .status)
  echo "  status: $STATE"
  case "$STATE" in
    done)   echo "→ $BASE_URL/jobs/$JOB/srt" ; break ;;
    failed) echo "job failed" >&2; exit 1 ;;
    *)      sleep 2 ;;
  esac
done

echo
echo "SRT:      $BASE_URL/jobs/$JOB/srt"
echo "TXT:      $BASE_URL/jobs/$JOB/txt"
echo "TIMINGS:  $BASE_URL/jobs/$JOB/timings"