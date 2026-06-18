#!/usr/bin/env bash
# Runs ON the vast.ai box (uploaded to /workspace/watchdog.sh, launched detached).
#
# LAST-RESORT BILLING BACKSTOP ONLY. It self-destructs the instance *only* if the box
# is still alive at the absolute DEADLINE (set to comfortably exceed the expected run
# time plus a generous reconnect window). It deliberately does NOT destroy the box when
# the run finishes: a completed box is kept alive on purpose so that a killed or
# disconnected monitor can be re-attached (`run_remote.sh monitor <IID> <RID>`) to pull
# the results before teardown. Normal teardown is the monitor's job — it destroys the
# box as soon as it has pulled the final results.
#
# This is the fix for "the box self-destructed and took un-pulled data with it": the
# only thing that auto-destroys a box now is genuine abandonment (nobody reconnects
# before the deadline), which prevents infinite orphan billing without ever racing the
# result-pull on a healthy completion.
#
#   bash watchdog.sh <INSTANCE_ID> <DEADLINE_EPOCH>
# Needs the vast API key at /root/.vk (staged by run_remote.sh).
set -u
IID="$1"; DEADLINE="$2"
while [ "$(date +%s)" -lt "$DEADLINE" ]; do sleep 60; done
echo "$(date -u +%FT%TZ) deadline reached -> destroying $IID (billing backstop)"
VK=$(cat /root/.vk 2>/dev/null)
for i in 1 2 3 4 5; do
  curl -s -X DELETE "https://console.vast.ai/api/v0/instances/$IID/?api_key=$VK" >/dev/null 2>&1 && { echo destroyed; exit 0; }
  sleep 10
done
echo "WARNING: self-destruct API call failed 5x"
