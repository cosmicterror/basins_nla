#!/usr/bin/env bash
# Runs ON the vast.ai box (uploaded to /workspace/watchdog.sh, launched detached).
#
# Self-destructs the instance via the vast.ai API so the box is NEVER left billing,
# regardless of what happens to the local launcher/monitor. It fires when:
#   - the run finishes  (GRACE seconds after the run writes $RUNDIR/DONE), OR
#   - an absolute DEADLINE passes (hard backstop if the run hangs or never starts).
# This is the core of the stability fix: box teardown does not depend on any
# process on David's machine staying alive.
#
#   bash watchdog.sh <INSTANCE_ID> <RUNDIR> <DEADLINE_EPOCH> <GRACE_SEC>
# Needs the vast API key at /root/.vk (staged by run_remote.sh).
set -u
IID="$1"; RUNDIR="$2"; DEADLINE="$3"; GRACE="$4"
reason=deadline
while true; do
  now=$(date +%s)
  [ "$now" -ge "$DEADLINE" ] && { reason=deadline; break; }
  if [ -f "$RUNDIR/DONE" ]; then
    d=$(cat "$RUNDIR/DONE" 2>/dev/null || echo 0)
    [ "$now" -ge $((d + GRACE)) ] && { reason=done; break; }
  fi
  sleep 30
done
echo "$(date -u +%FT%TZ) watchdog firing (reason=$reason) -> destroying $IID"
VK=$(cat /root/.vk 2>/dev/null)
for i in 1 2 3 4 5; do
  curl -s -X DELETE "https://console.vast.ai/api/v0/instances/$IID/?api_key=$VK" >/dev/null 2>&1 && { echo "destroyed"; exit 0; }
  sleep 10
done
echo "WARNING: self-destruct API call failed 5x"
