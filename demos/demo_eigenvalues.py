import argparse
from pathlib import Path
import sys
import numpy as np

# Allow running this demo directly from a source checkout.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jamgrad import Tensor


MATRIX_PRESETS = {
    "symmetric3": np.array(
        [
            [4.0, 1.0, -2.0],
            [1.0, 3.0, 0.0],
            [-2.0, 0.0, 2.0],
        ],
        dtype=np.float32,
    ),
    "spd4": np.array(
        [
            [6.0, 2.0, 1.0, 0.0],
            [2.0, 5.0, 2.0, 1.0],
            [1.0, 2.0, 4.0, 1.0],
            [0.0, 1.0, 1.0, 3.0],
        ],
        dtype=np.float32,
    ),
}


def power_iteration(matrix, num_iters=200, tol=1e-8, seed=42):
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Power iteration requires a square 2D matrix.")

    n = matrix.shape[0]
    rng = np.random.default_rng(seed)
    vector = Tensor(rng.normal(size=(n, 1)).astype(np.float32))
    vector = vector / np.linalg.norm(vector.data)

    prev_lambda = None
    lambda_est = 0.0
    step = 0

    for step in range(1, num_iters + 1):
        av = matrix @ vector
        norm_av = float(np.linalg.norm(av.data))
        if norm_av == 0.0:
            raise ValueError("Power iteration failed because Av became the zero vector.")

        vector = av / norm_av
        av = matrix @ vector

        numerator = float((vector * av).sum().data)
        denominator = float((vector * vector).sum().data)
        lambda_est = numerator / denominator

        if prev_lambda is not None and abs(lambda_est - prev_lambda) < tol:
            break
        prev_lambda = lambda_est

    return lambda_est, vector, step


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute dominant eigenvalue with jamgrad power iteration"
    )
    parser.add_argument(
        "--matrix",
        type=str,
        default="symmetric3",
        choices=sorted(MATRIX_PRESETS.keys()),
        help="Matrix preset to use",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=200,
        help="Maximum number of power-iteration steps (default: 200)",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-8,
        help="Convergence tolerance on eigenvalue change (default: 1e-8)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for initial vector"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    matrix_np = MATRIX_PRESETS[args.matrix]
    matrix = Tensor(matrix_np).set_label("A")

    dominant_est, eigenvector, steps = power_iteration(
        matrix, num_iters=args.iters, tol=args.tol, seed=args.seed
    )

    exact_eigenvalues = np.linalg.eigvals(matrix_np.astype(np.float64))
    exact_eigenvalues = np.real_if_close(exact_eigenvalues)
    dominant_exact = float(np.max(exact_eigenvalues))

    residual = np.linalg.norm(matrix_np @ eigenvector.data - dominant_est * eigenvector.data)
    abs_error = abs(dominant_est - dominant_exact)

    print("=== Eigenvalue Demo (jamgrad) ===")
    print(f"Preset: {args.matrix}")
    print("Matrix A:")
    print(matrix_np)
    print()
    print(f"Power iteration steps: {steps}")
    print(f"Estimated dominant eigenvalue: {dominant_est:.8f}")
    print(f"Exact dominant eigenvalue:     {dominant_exact:.8f}")
    print(f"Absolute error:                {abs_error:.3e}")
    print(f"Residual ||Av - lambda*v||:    {residual:.3e}")
    print()
    print("All eigenvalues from NumPy (reference):")
    print(np.sort(exact_eigenvalues)[::-1])
