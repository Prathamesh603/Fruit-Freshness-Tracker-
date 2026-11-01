# New detector workflow

Start with [the project start page](../../START_HERE.md). This is the only folder for the next detector experiment.

| Location | Purpose |
|---|---|
| `pretrained/` | Verified official YOLO26s starting weights and source/hash record; already downloaded |
| `configs/` | Target class order and, later, the actual reviewed training configuration |
| Root Python files, when added | Dataset preparation, training, evaluation and export entry points |
| `logs/`, when training starts | Console logs and epoch losses |
| `reports/`, when generated | Environment, data checks, metrics, class confusion and verification |
| `runs/`, when training starts | Best/last training checkpoints and plots |
| `artifacts/`, after verification | Exported deployment weights, labels, preprocessing and hashes |

Dataset home: [datasets/fruit9-v3](../../datasets/fruit9-v3/README.md). Do not make a second full dataset copy here unnecessarily. Logs, runs and deployment outputs are created when needed, not filled with placeholder results.

Current state: starting checkpoint downloaded; waiting for new dataset links. No training or app replacement has started. Keep the existing nine-fruit freshness classifier unchanged during detector work.
