# scripts/ — cloud deploy + smoke test

Shared infra (both repos carry a copy). Workflow: **prove the pipeline on a cheap
instance, then commit to an expensive one for real experiments.** Code stays local;
instances are create-and-destroy.

## Run order (on a vast.ai box)

```bash
# 0. local: launch a CHEAP Hopper box + upload, via the /gpu skill
#    (Hopper so the smoke test exercises the SAME fa3 path the H200 will use)

# 1. build env + apply the SGLang patches (the fragile, debug-prone step)
bash scripts/setup_remote.sh

# 2. serve the AV (verbaliser) checkpoint
bash scripts/serve_av.sh fa3            # Hopper (H100/H200). A100 -> use: triton
tail -f /workspace/sglang.log           # wait for "ready to roll"

# 3. validate injection: English out (PASS) vs CJK soup (FAIL)
python scripts/smoke_test.py --sglang-url http://localhost:30000
```

`smoke_test.py` exits 0 on pass, 1 on fail, and prints a checklist on failure.

## Why cheap-first

The GPU is not the risk — the **SGLang setup is**: `sglang[all]==0.5.6` can fight the
image's torch, the patch anchors can drift with version, Gemma is gated (needs the HF
token), and a wrong attention backend OOMs on `head_dim=256`. Debugging that at H200
rates is the waste we're avoiding. Smoke-test on a ~$1–2/hr Hopper; once green, the
exact same scripts run on the H200.

## Attention backend

| GPU | arch | backend | note |
|-----|------|---------|------|
| H100 / H200 | Hopper | `fa3` | documented happy path |
| A100 | Ampere | `triton` | fa3 is Hopper-only; triton also handles head_dim=256 |
| anything | — | ~~flashinfer~~ | the default — **OOMs on Gemma-3**, do not use |

## Tiers of the smoke test

- **default** (`smoke_test.py`): AV-only, random vectors. Fits one card at `mem_frac 0.85`.
  Validates server + patch + injection → English. This is the cheap pre-flight.
- **`--real`**: also verbalise a *true* Gemma L32 activation (loads base model ~24GB).
  Launch `serve_av.sh fa3 30000 0.5` so both fit. The genuine end-to-end check.
- **`--ar`**: also reconstruct + cosine-score (closes the autoencoder loop). Needs all
  three models resident — this is the H200 integration test, not the cheap pre-flight.
