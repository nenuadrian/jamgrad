# API Overview

`jamgrad` exposes a compact API centered on `Tensor` plus a small neural-network helper module.

## Import surface

```python
from jamgrad import Tensor
from jamgrad import nn
```

## `Tensor` (`jamgrad.tensor`)

Core features:

- Arithmetic with autograd support: `+`, `-`, `*`, `/`, `**`, unary `-`
- Matrix multiplication: `@`
- Unary ops: `.exp()`, `.log()`, `.sum()`
- Backpropagation entrypoint: `.backward()`
- Graph labeling and visualization: `.set_label()`, `.to_dot()`

## `nn` module (`jamgrad.nn`)

- `Linear(in_features, out_features)`
- `relu(x)`
- `softmax(x)`
- `cross_entropy_loss(predictions, targets)`

## Quick example

```python
from jamgrad import Tensor
from jamgrad.nn import Linear, relu

x = Tensor([[1.0, 2.0]], requires_grad=False)
layer = Linear(2, 1)
y = relu(layer(x))
```
