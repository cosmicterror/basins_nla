#!/usr/bin/env bash
# scripts/run_remote.sh — robust, self-healing vast.ai runner for basins_nla.
#
# WHY THIS EXISTS: earlier runs died not on the GPU but because the *local*
# orchestration process got killed (interface glitch / disconnect), and box
# teardown + result-saving were tied to that process. This runner removes that
# coupling. Three independent safety nets:
#   1. The experiment runs DETACHED on the box (setsid) -> survives SSH drops.
#   2. A self-destruct WATCHDOG is armed ON the box (_box_watchdog.sh) -> the box
#      tears itself down even if every local watcher dies. No orphan billing, ever.
#   3. Results are pulled INCREMENTALLY (every PULL_EVERY s) -> a killed monitor
#      loses at most the last few seconds of output.
# => Safe to Ctrl-C, safe to run as a Claude background task, safe to run in your
#    own terminal. Datacenter GPUs only (consumer cards were the other failure mode).
#
# USAGE
#   bash scripts/run_remote.sh launch  <RUN_ID> "<command>"   # provision+run+pull+destroy
#   bash scripts/run_remote.sh monitor <IID> <RUN_ID>         # re-attach if launcher died
#   bash scripts/run_remote.sh ps                             # what's billing right now
#   bash scripts/run_remote.sh kill    <IID>                  # destroy + verify
#
# The <command> must accept --run-id (arm_a.py / arm_b.py do); it is appended
# automatically so outputs land in runs/<RUN_ID>/ on the box.
# EXAMPLE (confirmatory §6.0 gate):
#   bash scripts/run_remote.sh launch gate03 "python src/arm_b.py --gate --repeats 2 --turns 18"
set -uo pipefail

# ── config (override via env) ──────────────────────────────────────────────────
VAST_KEY_FILE="${VAST_KEY_FILE:-$HOME/.config/vastai/vast_api_key}"   # same key the vastai CLI uses
SSH_KEY="${SSH_KEY:-$HOME/.ssh/vastai}"
IMAGE="${IMAGE:-pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime}"
DISK="${DISK:-50}"
GPU_RAM_MIN="${GPU_RAM_MIN:-44}"            # MB threshold passed to search (>=44GB fits Gemma-12B)
MAX_HOURS="${MAX_HOURS:-6}"                 # billing backstop: box self-destructs ONLY if still
# alive this long after launch (true abandonment/hang). It does NOT auto-destroy on completion —
# a finished box is KEPT so a killed/disconnected monitor can re-attach and pull results.
# Set it to comfortably exceed (run time + reconnect window). Healthy runs are torn down by the
# monitor the instant it pulls the final results, so the backstop rarely costs anything.
PULL_EVERY="${PULL_EVERY:-45}"            # incremental result pull cadence
PROJ="/workspace/basins_nla"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VAST_KEY="$(cat "$VAST_KEY_FILE" 2>/dev/null || true)"
[ -z "$VAST_KEY" ] && { printf '\033[31mFATAL: no vast.ai API key at %s (set one: vastai set api-key <KEY>)\033[0m\n' "$VAST_KEY_FILE" >&2; exit 1; }
DC_RE='^(A100|A800|H100|H200|H800|L40|L40S|A40)'   # datacenter GPU-model whitelist (see feedback-vast-gpu-choice)
BAD_MACHINES_FILE="${BAD_MACHINES_FILE:-$HOME/.config/vastai/bad_machines}"  # machine_ids that crashed; skipped on future launches

SSHOPTS=(-i "$SSH_KEY" -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null
         -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -o ConnectTimeout=20 -o LogLevel=ERROR)

log(){ printf '\033[36m[%s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*"; }
err(){ printf '\033[31m[%s] %s\033[0m\n' "$(date +%H:%M:%S)" "$*" >&2; }

# ── vast API helpers (python: robust JSON, no jq dependency) ────────────────────
list_instances(){ python3 - "$VAST_KEY" <<'PY'
import sys,urllib.request,urllib.parse,json
try:
    d=json.load(urllib.request.urlopen("https://console.vast.ai/api/v0/instances/?"+
        urllib.parse.urlencode({"api_key":sys.argv[1],"owner":"me"}),timeout=60)).get("instances",[])
    for i in d: print(i["id"], i.get("actual_status"), (i.get("gpu_name") or "").replace(" ","_"), i.get("dph_total"))
except Exception as e:
    print("LISTERR", e, file=sys.stderr); sys.exit(3)
PY
}
destroy_instance(){ python3 - "$1" "$VAST_KEY" <<'PY'
import sys,urllib.request,urllib.parse,json
iid,key=sys.argv[1],sys.argv[2]
req=urllib.request.Request("https://console.vast.ai/api/v0/instances/%s/?%s"%(iid,
    urllib.parse.urlencode({"api_key":key})),method="DELETE")
try: print("destroy",iid,"->",json.load(urllib.request.urlopen(req,timeout=60)).get("success"))
except Exception as e: print("destroy-err",iid,e)
PY
}
record_bad(){  # machine_id -> append to the persistent bad-machine list (skipped on future launches)
  local mid="$1"
  [ -z "$mid" ] && return 0
  [ "$mid" = "None" ] && return 0
  mkdir -p "$(dirname "$BAD_MACHINES_FILE")"
  grep -qxF "$mid" "$BAD_MACHINES_FILE" 2>/dev/null && return 0
  echo "$mid" >> "$BAD_MACHINES_FILE"
  log "machine $mid added to bad-list ($BAD_MACHINES_FILE)" >&2
}
verify_clean(){
  local out rc; out=$(list_instances 2>/dev/null); rc=$?
  if [ "$rc" -ne 0 ]; then err "⚠️ could NOT verify instance state (API error) — CHECK MANUALLY: vastai show instances"; return; fi
  if [ -z "$out" ]; then log "✅ no instances on the account — nothing billing"; else
    err "⚠️ instances still present:"; printf '%s\n' "$out" >&2
    err "   destroy with: bash scripts/run_remote.sh kill <id>"; fi
}

# ── provisioning ────────────────────────────────────────────────────────────────
pick_candidates(){  # N -> lines "offer_id machine_id gpu_name dph"  (TRUE datacenter hosts only)
  vastai search offers \
    "datacenter=true num_gpus=1 gpu_ram>=${GPU_RAM_MIN} disk_space>=${DISK} reliability>0.98 rentable=true verified=true cuda_vers>=12.1 inet_down>=200" \
    -o 'dph+' --raw 2>/dev/null | python3 -c "
import sys,json,re
N=int('${1:-4}'); dc=re.compile('${DC_RE}'); out=[]
for o in json.load(sys.stdin):
    gn=(o.get('gpu_name') or '').replace(' ','_')
    # hosting_type==1 is a true datacenter host (0 = community/home rig); belt-and-suspenders to datacenter=true
    if dc.match(gn) and (o.get('inet_down') or 0)>=1000 and o.get('hosting_type')==1:
        out.append('%s %s %s %.3f'%(o['id'],o.get('machine_id'),gn,o.get('dph_total',0)))
print('\n'.join(out[:N]))
"
}
list_ids(){ python3 - "$VAST_KEY" <<'PY'
import sys,urllib.request,urllib.parse,json
try:
    d=json.load(urllib.request.urlopen("https://console.vast.ai/api/v0/instances/?"+
        urllib.parse.urlencode({"api_key":sys.argv[1],"owner":"me"}),timeout=60)).get("instances",[])
    for i in d: print(i["id"])
except Exception: sys.exit(3)
PY
}
create(){  # offer_id -> iid
  # NB: `vastai create --raw` prints nothing in this CLI version but DOES create the
  # box. So we detect the new instance by diffing the id list before/after — this also
  # catches a "looks-failed-but-actually-created" box instead of spawning duplicates.
  local off="$1" before after newid
  before=$(list_ids 2>/dev/null) || return 1
  vastai create instance "$off" --image "$IMAGE" --disk "$DISK" --ssh >/dev/null 2>&1
  for _ in 1 2 3 4 5 6 7; do
    sleep 3
    after=$(list_ids 2>/dev/null) || continue
    newid=$(comm -13 <(printf '%s\n' "$before" | sort -u) <(printf '%s\n' "$after" | sort -u) | grep -E '^[0-9]+$' | head -1)
    [ -n "$newid" ] && { echo "$newid"; return 0; }
  done
  return 1
}
ssh_hostport(){  # iid -> "host port"
  local url; url=$(vastai ssh-url "$1" 2>/dev/null)
  [[ "$url" =~ ^ssh://root@([^:]+):([0-9]+) ]] && echo "${BASH_REMATCH[1]} ${BASH_REMATCH[2]}"
}
wait_ssh(){  # iid -> 0 when ssh answers
  local iid="$1" hp host port
  for _ in $(seq 1 45); do
    hp=$(ssh_hostport "$iid"); host="${hp% *}"; port="${hp#* }"
    if [ -n "$hp" ] && ssh "${SSHOPTS[@]}" -p "$port" "root@$host" true 2>/dev/null; then return 0; fi
    sleep 10
  done; return 1
}
cuda_ok(){  # host port
  ssh "${SSHOPTS[@]}" -p "$2" "root@$1" \
    "python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)'" 2>/dev/null
}

# ── staging / launch ────────────────────────────────────────────────────────────
stage_keys(){  # host port
  local hftok anth
  hftok=$(cat "$HOME/.cache/huggingface/token" 2>/dev/null)
  anth=$(python3 -c "import os,re;m=re.search(r'ANTHROPIC_API_KEY=[\"\x27]?([A-Za-z0-9_\-]+)',open(os.path.expanduser('~/.zshrc')).read());print(m.group(1) if m else '')")
  ssh "${SSHOPTS[@]}" -p "$2" "root@$1" "umask 077; cat > /root/.runenv" <<EOF
export HF_TOKEN='$hftok'
export ANTHROPIC_API_KEY='$anth'
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHONUNBUFFERED=1
EOF
  ssh "${SSHOPTS[@]}" -p "$2" "root@$1" "umask 077; cat > /root/.vk" <<<"$VAST_KEY"
}
arm_watchdog(){  # iid host port run_id
  local iid="$1" host="$2" port="$3" rid="$4"
  local rundir="$PROJ/runs/$rid" deadline=$(( $(date +%s) + MAX_HOURS*3600 ))
  scp "${SSHOPTS[@]}" -P "$port" "$REPO/scripts/_box_watchdog.sh" "root@$host:/workspace/watchdog.sh" >/dev/null 2>&1
  ssh "${SSHOPTS[@]}" -p "$port" "root@$host" \
    "mkdir -p '$rundir'; setsid bash /workspace/watchdog.sh $iid $deadline >/workspace/watchdog.log 2>&1 </dev/null & echo ARMED"
  log "billing backstop armed: box self-destructs ONLY if still alive in ${MAX_HOURS}h (NOT on completion — a finished box is kept for re-attach)"
}
setup_box(){  # host port
  log "installing deps on box (transformers==4.57.1, accelerate, anthropic, hf_transfer)..."
  ssh "${SSHOPTS[@]}" -p "$2" "root@$1" \
    "pip install -q 'transformers==4.57.1' accelerate anthropic pyyaml safetensors numpy hf_transfer 2>&1 | tail -1"
}
stage_code(){  # host port
  log "uploading code (src/ configs/ data/ scripts/)..."
  COPYFILE_DISABLE=1 tar czf - -C "$REPO" src configs data scripts 2>/dev/null | \
    ssh "${SSHOPTS[@]}" -p "$2" "root@$1" "mkdir -p $PROJ && tar xzf - -C $PROJ 2>/dev/null"
}
launch_detached(){  # host port run_id cmd
  local host="$1" port="$2" rid="$3" cmd="$4" rundir="$PROJ/runs/$3"
  ssh "${SSHOPTS[@]}" -p "$port" "root@$host" \
    "source /root/.runenv; mkdir -p '$rundir'; setsid bash -c 'cd $PROJ; ($cmd --run-id $rid > \"$rundir/run.log\" 2>&1; echo \$? > \"$rundir/EXIT\"; date +%s > \"$rundir/DONE\")' >/dev/null 2>&1 </dev/null & echo LAUNCHED"
  log "experiment launched detached on box"
}

# ── monitoring ──────────────────────────────────────────────────────────────────
pull_once(){  # host port run_id  (tar over ssh; tolerant of transient failure)
  ssh "${SSHOPTS[@]}" -p "$2" "root@$1" "cd $PROJ && tar czf - runs/$3 2>/dev/null" 2>/dev/null \
    | tar xzf - -C "$REPO" 2>/dev/null || true
}
monitor_loop(){  # iid host port run_id
  local iid="$1" host="$2" port="$3" rid="$4" L="$REPO/runs/$4" gone=0
  while true; do
    pull_once "$host" "$port" "$rid"
    if [ -f "$L/DONE" ]; then log "DONE detected — run finished"; break; fi
    local inst; inst=$(list_instances 2>/dev/null)
    if [ $? -eq 0 ] && ! grep -q "^$iid " <<<"$inst"; then
      gone=$((gone+1)); [ $gone -ge 2 ] && { err "instance $iid gone (watchdog fired) — stopping monitor"; break; }
    else gone=0; fi
    local last; last=$(tail -n 1 "$L/run.log" 2>/dev/null); [ -n "$last" ] && printf '    %s\n' "$last"
    sleep "$PULL_EVERY"
  done
}
report(){  # run_id
  local L="$REPO/runs/$1"
  echo "════════════════════ RESULT: $1 ════════════════════"
  echo "exit code: $(cat "$L/EXIT" 2>/dev/null || echo '? (run cut before completion)')"
  [ -f "$L/gate_report.md" ] && { echo "──── gate_report.md ────"; cat "$L/gate_report.md"; }
  echo "──── tail run.log ────"; tail -n 12 "$L/run.log" 2>/dev/null
  echo "local results: runs/$1/"
}

# ── top-level commands ──────────────────────────────────────────────────────────
provision_box(){  # $1 = extra (session) machine_ids to skip; echoes "iid host port machine_id gpu"
  local sess=" ${1:-} " bad=" " CANDS=() line off mid gpu dph iid hp host port
  [ -f "$BAD_MACHINES_FILE" ] && bad=" $(tr '\n' ' ' < "$BAD_MACHINES_FILE") "
  while IFS= read -r line; do [ -n "$line" ] && CANDS+=("$line"); done < <(pick_candidates 12)
  for line in "${CANDS[@]}"; do
    set -- $line; off="$1"; mid="$2"; gpu="$3"; dph="$4"
    case "$sess$bad" in *" $mid "*) log "skip offer $off (machine $mid on bad-list)" >&2; continue ;; esac
    log "trying offer $off  (machine $mid, $gpu, \$$dph/hr)" >&2
    iid=$(create "$off"); [ -z "$iid" ] && { err "create failed; next" >&2; continue; }
    log "instance $iid created; waiting for SSH (up to ~7m)..." >&2
    if ! wait_ssh "$iid"; then err "no SSH; destroying $iid" >&2; destroy_instance "$iid" >/dev/null 2>&1; continue; fi
    hp=$(ssh_hostport "$iid"); host="${hp% *}"; port="${hp#* }"
    if ! cuda_ok "$host" "$port"; then err "CUDA bad on $iid (machine $mid); marking bad + destroying" >&2; record_bad "$mid"; destroy_instance "$iid" >/dev/null 2>&1; continue; fi
    log "box $iid ready: $gpu @ root@$host:$port (machine $mid, CUDA ok)" >&2
    echo "$iid $host $port $mid $gpu"; return 0
  done
  return 1
}

launch_main(){
  local rid="${1:-}"; shift || true; local cmd="${*:-}"
  [ -z "$rid" ] || [ -z "$cmd" ] && { err 'usage: launch <RUN_ID> "<command>"'; exit 2; }
  rm -rf "$REPO/runs/$rid"; mkdir -p "$REPO/runs/$rid"      # fresh launch (use `monitor` to re-attach a partial)
  local skip="" attempt=0 maxr="${MAX_RUN_RETRIES:-4}"
  while [ "$attempt" -lt "$maxr" ]; do
    attempt=$((attempt+1))
    local box; box=$(provision_box "$skip") || { err "no usable datacenter box left to try"; break; }
    set -- $box; local iid="$1" host="$2" port="$3" mid="$4" gpu="$5"
    printf 'IID=%s\nHOST=%s\nPORT=%s\nGPU=%s\n' "$iid" "$host" "$port" "$gpu" > "$REPO/runs/$rid/.instance"
    stage_keys "$host" "$port"
    arm_watchdog "$iid" "$host" "$port" "$rid"     # protected from here on, no matter what
    setup_box "$host" "$port"
    stage_code "$host" "$port"
    launch_detached "$host" "$port" "$rid" "$cmd"
    log "attempt $attempt/$maxr — monitoring (pull every ${PULL_EVERY}s). Safe to Ctrl-C / disconnect; box keeps results."
    log "** if this monitor is killed, RE-ATTACH: bash scripts/run_remote.sh monitor $iid $rid **"
    monitor_loop "$iid" "$host" "$port" "$rid"
    pull_once "$host" "$port" "$rid"
    local ec nres
    ec=$(cat "$REPO/runs/$rid/EXIT" 2>/dev/null || echo "?")
    nres=$(ls "$REPO/runs/$rid/transcripts" 2>/dev/null | wc -l | tr -d ' ')
    destroy_instance "$iid"; sleep 3
    if [ "$ec" = "0" ] || [ "${nres:-0}" -gt 0 ]; then   # success, or real progress worth keeping
      verify_clean; report "$rid"; return 0
    fi
    err "attempt $attempt: machine $mid (inst $iid) crashed early (exit=$ec, 0 results) — marking bad + retrying on a fresh machine"
    record_bad "$mid"; skip="$skip $mid"
    rm -f "$REPO/runs/$rid/EXIT" "$REPO/runs/$rid/DONE" "$REPO/runs/$rid/run.log"
  done
  err "giving up after $attempt attempt(s) — no run produced results"; verify_clean; exit 1
}

cmd="${1:-}"; shift 2>/dev/null || true
case "$cmd" in
  launch)  launch_main "$@" ;;
  monitor) iid="${1:-}"; rid="${2:-}"; [ -z "$iid" ] || [ -z "$rid" ] && { err "usage: monitor <IID> <RUN_ID>"; exit 2; }
           hp=$(ssh_hostport "$iid"); host="${hp% *}"; port="${hp#* }"
           [ -z "$hp" ] && { err "can't resolve SSH for $iid"; exit 1; }
           monitor_loop "$iid" "$host" "$port" "$rid"; pull_once "$host" "$port" "$rid"
           destroy_instance "$iid"; sleep 5; verify_clean; report "$rid" ;;
  ps)      out=$(list_instances 2>/dev/null); rc=$?
           if [ "$rc" -ne 0 ]; then err "API error — could not list instances"; exit 1
           elif [ -z "$out" ]; then echo "(no instances — nothing billing)"; else printf '%s\n' "$out"; fi ;;
  kill)    [ -z "${1:-}" ] && { err "usage: kill <IID>"; exit 2; }; destroy_instance "$1"; sleep 3; verify_clean ;;
  *)       echo 'usage: run_remote.sh {launch <RUN_ID> "<cmd>" | monitor <IID> <RUN_ID> | ps | kill <IID>}'; exit 2 ;;
esac
