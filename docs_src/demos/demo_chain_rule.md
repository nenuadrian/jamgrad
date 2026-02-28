# `demo_chain_rule.py`

This demo focuses on chain-rule behavior across increasingly complex expressions.

## Run

```bash
python demos/demo_chain_rule.py
```

## What it covers

- Simple composition: `f(g(x))`
- Triple composition: `f(g(h(x)))`
- Product rule combined with chain rule
- Exponential of a composed function
- Power of an affine transform
- Multivariable chain rule for `f(x, y)`

Each example compares autograd output with a hand-derived expected value and asserts numerical agreement.

## Output artifacts

The script writes a DOT graph for the final example:

```text
chain_rule_graph.dot
```

To render it:

```bash
dot -Tpng chain_rule_graph.dot -o chain_rule_graph.png
```
