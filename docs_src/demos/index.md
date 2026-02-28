# Demos

`jamgrad` includes five runnable demos that exercise core tensor ops, backpropagation, and neural-network training.

| Script | What it shows | Run |
| --- | --- | --- |
| `demos/demo.py` | Core tensor ops, gradients, and graph export | `python demos/demo.py` |
| `demos/demo_chain_rule.py` | Chain-rule walkthrough with assertions | `python demos/demo_chain_rule.py` |
| `demos/demo_eigenvalues.py` | Power iteration with `Tensor` matmul | `python demos/demo_eigenvalues.py --matrix symmetric3 --iters 200` |
| `demos/demo_nn.py` | XOR training with a small MLP | `python demos/demo_nn.py --epochs 1000 --lr 0.5` |
| `demos/demo_mnist.py` | MNIST classifier training | `python demos/demo_mnist.py --dataset-size 5000 --epochs 20` |

Use the pages in this section for full options and output details per demo.
