import numpy as np

from jamgrad import Tensor


# Basic tensor operations
print("=== Basic Tensor Operations ===")
x = Tensor([1, 2, 3])
y = Tensor([4, 5, 6])
z = x + y
print(f"x: {x}")
print(f"y: {y}")
print(f"x + y: {z}")

# Gradient computation
print("\n=== Gradient Computation ===")
x = Tensor([2.0], requires_grad=True)
y = x**2
print(f"x: {x}")
print(f"y = x^2: {y}")

y.backward()
print(f"dy/dx: {x.grad}")

# Chain rule example
print("\n=== Chain Rule Example ===")
x = Tensor([3.0], requires_grad=True)
y = x**2
z = y * 2 + 1
print(f"x: {x}")
print(f"z = 2x^2 + 1: {z}")

z.backward()
print(f"dz/dx: {x.grad}")

# Multiple operations
print("\n=== Complex Expression ===")
x = Tensor([1.0], requires_grad=True)
y = Tensor([2.0], requires_grad=True)
z = (x**2 + y) * x.exp() + y.log()
print(f"x: {x}")
print(f"y: {y}")
print(f"z = (x^2 + y) * exp(x) + log(y): {z}")

z.backward()
print(f"dz/dx: {x.grad}")
print(f"dz/dy: {y.grad}")
