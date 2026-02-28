# `demo_nn.py`

This demo trains a two-layer neural network on XOR.

## Run

```bash
python demos/demo_nn.py --epochs 1000 --lr 0.5 --hidden-size 4
```

Quiet smoke run (useful for CI):

```bash
python demos/demo_nn.py --epochs 20 --log-interval 20 --quiet
```

## CLI options

| Option | Default | Description |
| --- | --- | --- |
| `--epochs` | `1000` | Training epochs |
| `--lr` | `0.5` | Learning rate |
| `--hidden-size` | `4` | Hidden layer width |
| `--log-interval` | `100` | Print every N epochs |
| `--seed` | `42` | Random seed |
| `--save-graph` | `None` | Save final prediction graph to `<name>.dot` |
| `--quiet` | `False` | Suppress console logs |

## Output

- Training-loss progression (unless `--quiet`)
- Final XOR predictions for 4 input pairs
- Optional DOT graph file when `--save-graph` is used
