# Public dataset (`sma-dataset`)

A shareable record of what was learned: your matrices, every configuration benchmarked on them, per GPU,
and the best one. It is **separate from campaigns**: a campaign never writes to it; you add a campaign's
results when you choose, and publish when you choose. The tool is application-agnostic: you bring your
matrices (MatrixMarket or PETSc binary); nothing here generates them.

Published instance: <https://huggingface.co/datasets/moletteremi/solver-market-agentic-database> (CC-BY-4.0).
Local copy: `$SMA_WORK/dataset` (every command takes `--dir DIR` to use another one).

## What it contains

No hypotheses, comments or reasoning: decks and measurements only.

| file | content |
|---|---|
| `best.jsonl` | best valid configuration per (matrix, GPU, k, tol): `time_per_solve_ms`, `iterations`, `setup_ms`, deck |
| `runs.jsonl` | every configuration run, valid or not (`error` says why), same readable columns first |
| `matrices.jsonl`, `matrices/<name>/` | `A.mtx.gz` (MatrixMarket), `A.csr` + `b.vec` (binary, `sma/csr_io.py`), `info.json` (origin, blocks, analysis) |
| `best/<matrix>/<gpu>/k<k>_tol<tol>.yaml` | the best deck, runnable with `sma-tools run` |
| `README.md` | dataset card (leaderboard, matrices, how configurations are measured), generated |

- Matrices are identified by the name you give; a matrix already present (same binary content) is not
  added twice, its entry gets the new origin in `also_from`.
- The best configuration is the lowest valid cost (`time_per_solve_ms` = (t_setup + k t_solve) / k, k = 0:
  solve time only), whatever the node; repeats of the same deck are listed.
- Local paths (account, project, build directories) are stripped from everything published.

## Commands

```bash
source machines/jean-zay/env.sh

# 1. add your matrix: one CPU job converts + analyzes it into $SMA_WORK/data/NAME and packages it
sma-dataset add-matrix my_matrix.mtx --name my_case_1M \
    --origin "my code, 3D pressure Poisson, P1 FEM, 1M unknowns" \
    --blocks "u: rows 0-999999"                        # optional; --rhs ones|zeros|FILE (default ones)

# 2. benchmark it with a campaign (README), using the converted system $SMA_WORK/data/my_case_1M
sma campaign $SMA_WORK/campaigns/my_case_1M-k0 --system $SMA_WORK/data/my_case_1M --budget 40

# 3. add the campaign's results, check, publish
sma-dataset add-campaign $SMA_WORK/campaigns/my_case_1M-k0
sma-dataset build                                      # prints the best per matrix / GPU
sma-dataset publish                                    # uploads to $SMA_HF_REPO (machines/<machine>/env.sh)
```

Supported matrix files: MatrixMarket coordinate (`.mtx`), PETSc binary AIJ with 32- or 64-bit indices
(a Vec stored after the matrix in the same file is ignored).

## Publishing (Hugging Face)

Once, store a write token for your account:
```bash
module load pytorch-gpu/py3/2.8.0 && HF_HOME=~/.cache/huggingface hf auth login
```
`sma-dataset publish` rebuilds the tables, refuses to upload if a published text file still contains a
local path, then runs `hf upload-large-folder` (resumable). On Jean Zay the `hf` CLI comes from the module
in `SMA_HF_SETUP` (machines/jean-zay/env.sh), and `HF_HOME=~/.cache/huggingface` is set explicitly: without
it, `hf` does not find the token. Proxy errors ("403 Forbidden") are transient; publish retries 3 times.

| symptom | what to do |
|---|---|
| `hf`: "Not logged in" | `module load pytorch-gpu/py3/2.8.0; HF_HOME=~/.cache/huggingface hf auth whoami` |
| publish fails after the retries | rerun later (the upload resumes) |
