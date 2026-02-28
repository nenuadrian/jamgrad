# `demo_mnist.py`

This demo trains a three-layer neural network on an MNIST subset using `jamgrad` primitives.

## Run

```bash
python demos/demo_mnist.py --dataset-size 5000 --epochs 20 --batch-size 64 --lr 0.01
```

Fast sanity run:

```bash
python demos/demo_mnist.py --dataset-size 1000 --epochs 1 --batch-size 128 --sample-predictions 0 --quiet
```

## Requirements

- `scikit-learn` (for dataset loading and preprocessing)
- Network access to fetch MNIST from OpenML on first run

## CLI options

| Option | Default | Description |
| --- | --- | --- |
| `--dataset-size` | `5000` | Number of MNIST samples to use |
| `--test-split` | `0.2` | Fraction reserved for test set |
| `--hidden1` | `128` | First hidden layer width |
| `--hidden2` | `64` | Second hidden layer width |
| `--epochs` | `100` | Training epochs |
| `--batch-size` | `32` | Batch size |
| `--lr` | `0.01` | Learning rate |
| `--grad-clip` | `1.0` | Gradient clipping threshold |
| `--log-interval` | `10` | Print every N epochs |
| `--seed` | `42` | Random seed |
| `--save-graph` | `None` | Save final test prediction graph to `<name>.dot` |
| `--quiet` | `False` | Suppress console logs |
| `--sample-predictions` | `10` | Number of test samples for final prediction pass |

## Output

- Training loss and test accuracy checkpoints (unless `--quiet`)
- Best observed test accuracy summary
- Optional DOT graph file when `--save-graph` is provided
