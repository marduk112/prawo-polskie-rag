#!/bin/bash
# Strażnik kosztów (Krok 76): zatrzymuje poda (stop -- /workspace zostaje), gdy
#  - praca skończona (ALL_DONE/FAILED) i Mac nie usunął poda w ciągu 60 min, albo
#  - pod działa dłużej niż MAX_HOURS (twardy limit budżetu).
# Wymaga RUNPOD_API_KEY i RUNPOD_POD_ID w środowisku poda.
MAX_MIN=${MAX_MIN:-600}  # twardy limit w minutach (bash nie liczy ułamków godzin)
W=/workspace/work
start=$(date +%s)
log() { echo "$(date -u +%H:%M:%S) $*" >> $W/logs/watchdog.log; }
stop_pod() {
  log "STOP: $1"
  curl -s -X POST "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID/stop" -H "Authorization: Bearer $RUNPOD_API_KEY" >> $W/logs/watchdog.log 2>&1
  sleep 600
}
log "start, limit ${MAX_MIN} min"
while true; do
  now=$(date +%s)
  for f in ALL_DONE FAILED; do
    if [ -f $W/$f ] && [ $(( now - $(stat -c %Y $W/$f) )) -gt 3600 ]; then stop_pod "$f od ponad 60 min"; fi
  done
  [ $(( now - start )) -gt $(( MAX_MIN * 60 )) ] && stop_pod "limit ${MAX_MIN} min"
  sleep 60
done
