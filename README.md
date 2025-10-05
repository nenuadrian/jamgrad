# jamgrad

![logo](./assets/jamgrad.png)

## setup
```bash
python -m venv jamenv
source jamenv/bin/activate

pip install pytest pytest-cov numpy torch
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
