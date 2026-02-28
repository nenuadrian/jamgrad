# `demo_eigenvalues.py`

This demo computes a dominant eigenvalue using power iteration implemented with `Tensor` operations.

## Run

```bash
python demos/demo_eigenvalues.py --matrix symmetric3 --iters 200
```

## CLI options

| Option | Default | Description |
| --- | --- | --- |
| `--matrix` | `symmetric3` | Matrix preset (`symmetric3`, `spd4`) |
| `--iters` | `200` | Maximum iteration count |
| `--tol` | `1e-8` | Convergence tolerance on eigenvalue change |
| `--seed` | `42` | Random seed for initial vector |

## Output

The script prints:

- Chosen matrix preset and matrix values
- Iteration count used
- Estimated dominant eigenvalue
- Exact dominant eigenvalue from NumPy
- Absolute error
- Residual norm `||Av - lambda*v||`
