# `demo.py`

This is the basic end-to-end tensor and autograd demo.

## Run

```bash
python demos/demo.py
```

## What it covers

- Tensor creation and arithmetic (`+`, `*`, `**`)
- Scalar backpropagation with `backward()`
- A composed expression:

$$
z = (x^2 + y)e^x + \ln(y)
$$

- Gradient inspection for both `x` and `y`
- Computation graph export in DOT format

## Output artifacts

The script writes:

```text
computation_graph.dot
```

To render it:

```bash
dot -Tpng computation_graph.dot -o assets/computation_graph.png
```
