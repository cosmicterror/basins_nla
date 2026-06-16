#!/usr/bin/env bash
# Launch the SGLang server for the Gemma-3-12B AV (verbaliser) checkpoint, backgrounded.
#
#   bash scripts/serve_av.sh [BACKEND] [PORT] [MEM_FRAC] [CKPT]
#     BACKEND  : fa3 (Hopper: H100/H200) | triton (Ampere: A100). default fa3
#                flashinfer (the default) OOMs on Gemma-3 head_dim=256 — don't use it.
#     PORT     : default 30000
#     MEM_FRAC : SGLang static mem fraction. default 0.85 (AV-only).
#                Lower to ~0.5 if you also need base Gemma + AR resident on one card.
#     CKPT     : default kitft/nla-gemma3-12b-L32-av
#
# Watch /workspace/sglang.log for "The server is fired up and ready to roll".
set -euo pipefail
BACKEND="${1:-fa3}"
PORT="${2:-30000}"
MEM_FRAC="${3:-0.85}"
CKPT="${4:-kitft/nla-gemma3-12b-L32-av}"
LOG=/workspace/sglang.log

echo "launching SGLang: ckpt=$CKPT backend=$BACKEND port=$PORT mem_frac=$MEM_FRAC"
nohup python -m sglang.launch_server \
    --model-path "$CKPT" \
    --port "$PORT" \
    --disable-radix-cache \
    --attention-backend "$BACKEND" \
    --mem-fraction-static "$MEM_FRAC" \
    --trust-remote-code \
    > "$LOG" 2>&1 &
echo $! > /workspace/sglang.pid
echo "pid $(cat /workspace/sglang.pid) -> $LOG"
echo "tail -f $LOG   # wait for 'ready to roll' before running smoke_test.py"
