# jamgrad

![logo](./assets/jamgrad.png)

Inspired by [Micrograd](https://github.com/karpathy/micrograd/tree/master) and many other light autograd implementations.

But why jamgrad? Because it's jam-packed with features! Actually, no, I am from Romania, we love making jam.

## setup
```bash
python -m venv jamenv
source jamenv/bin/activate

pip install pytest pytest-cov numpy torch scikit-learn pandas
```

## test

To run tests with verbose output:
```bash
pytest tests -v
```

With coverage:
```bash
pytest tests --cov=jamgrad
```
