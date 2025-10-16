from jamgrad import Tensor


print("=" * 60)
print("CHAIN RULE DEMONSTRATIONS WITH JAMGRAD")
print("=" * 60)

# Example 1: Simple composition f(g(x))
print("\n=== Example 1: Simple Composition ===")
print("Function: f(u) = u^2 where u = g(x) = 3x")
print("Expected: df/dx = 2u * 3 = 6u = 18x")

x = Tensor([2.0], requires_grad=True).set_label("x")
u = 3 * x  # g(x) = 3x
u.set_label("u")
f = u**2  # f(u) = u^2
f.set_label("f")

print(f"\nx = {x.data[0]}")
print(f"u = 3x = {u.data[0]}")
print(f"f = u^2 = {f.data[0]}")

f.backward()
print(f"\ndf/dx = {x.grad.data[0]}")
print(f"Expected: 18 * {x.data[0]} = {18 * x.data[0]}")

# Example 2: Nested functions f(g(h(x)))
print("\n\n=== Example 2: Triple Composition ===")
print("Function: f(v) = v^2 where v = g(u) = u + 1 where u = h(x) = 2x")
print("Expected: df/dx = 2v * 1 * 2 = 4v")

x = Tensor([3.0], requires_grad=True).set_label("x")
u = 2 * x  # h(x) = 2x
u.set_label("u")
v = u + 1  # g(u) = u + 1
v.set_label("v")
f = v**2  # f(v) = v^2
f.set_label("f")

print(f"\nx = {x.data[0]}")
print(f"u = 2x = {u.data[0]}")
print(f"v = u + 1 = {v.data[0]}")
print(f"f = v^2 = {f.data[0]}")

f.backward()
print(f"\ndf/dx = {x.grad.data[0]}")
print(f"Expected: 4 * {v.data[0]} = {4 * v.data[0]}")

# Example 3: Product rule combined with chain rule
print("\n\n=== Example 3: Product Rule + Chain Rule ===")
print("Function: f(x) = x^2 * (2x + 1)")
print("Expected: df/dx = 2x(2x+1) + x^2(2) = 4x^2 + 2x + 2x^2 = 6x^2 + 2x")

x = Tensor([2.0], requires_grad=True).set_label("x")
u = x**2
u.set_label("u=x^2")
v = 2 * x + 1
v.set_label("v=2x+1")
f = u * v
f.set_label("f")

print(f"\nx = {x.data[0]}")
print(f"u = x^2 = {u.data[0]}")
print(f"v = 2x + 1 = {v.data[0]}")
print(f"f = u * v = {f.data[0]}")

f.backward()
expected = 6 * x.data[0]**2 + 2 * x.data[0]
print(f"\ndf/dx = {x.grad.data[0]}")
print(f"Expected: 6({x.data[0]})^2 + 2({x.data[0]}) = {expected}")

# Example 4: Exponential and logarithm chain rule
print("\n\n=== Example 4: Exponential Chain Rule ===")
print("Function: f(x) = exp(x^2)")
print("Expected: df/dx = exp(x^2) * 2x")

x = Tensor([1.5], requires_grad=True).set_label("x")
u = x**2
u.set_label("u=x^2")
f = u.exp()
f.set_label("f=exp(u)")

print(f"\nx = {x.data[0]}")
print(f"u = x^2 = {u.data[0]}")
print(f"f = exp(u) = {f.data[0]}")

f.backward()
import math
expected = math.exp(x.data[0]**2) * 2 * x.data[0]
print(f"\ndf/dx = {x.grad.data[0]}")
print(f"Expected: exp({x.data[0]}^2) * 2 * {x.data[0]} = {expected}")

# Example 5: Power rule with chain rule
print("\n\n=== Example 5: Power of Composition ===")
print("Function: f(x) = (2x + 3)^3")
print("Expected: df/dx = 3(2x + 3)^2 * 2")

x = Tensor([1.0], requires_grad=True).set_label("x")
u = 2 * x + 3
u.set_label("u=2x+3")
f = u**3
f.set_label("f=u^3")

print(f"\nx = {x.data[0]}")
print(f"u = 2x + 3 = {u.data[0]}")
print(f"f = u^3 = {f.data[0]}")

f.backward()
expected = 3 * (2 * x.data[0] + 3)**2 * 2
print(f"\ndf/dx = {x.grad.data[0]}")
print(f"Expected: 3 * ({u.data[0]})^2 * 2 = {expected}")

# Example 6: Multiple variables with chain rule
print("\n\n=== Example 6: Multivariable Chain Rule ===")
print("Function: f(x, y) = (x*y)^2")
print("Expected: df/dx = 2(xy) * y, df/dy = 2(xy) * x")

x = Tensor([2.0], requires_grad=True).set_label("x")
y = Tensor([3.0], requires_grad=True).set_label("y")
u = x * y
u.set_label("u=xy")
f = u**2
f.set_label("f=u^2")

print(f"\nx = {x.data[0]}, y = {y.data[0]}")
print(f"u = xy = {u.data[0]}")
print(f"f = u^2 = {f.data[0]}")

f.backward()
expected_dx = 2 * (x.data[0] * y.data[0]) * y.data[0]
expected_dy = 2 * (x.data[0] * y.data[0]) * x.data[0]
print(f"\ndf/dx = {x.grad.data[0]}")
print(f"Expected: 2 * {u.data[0]} * {y.data[0]} = {expected_dx}")
print(f"df/dy = {y.grad.data[0]}")
print(f"Expected: 2 * {u.data[0]} * {x.data[0]} = {expected_dy}")

# Save computation graph for the last example
print("\n\n=== Computation Graph ===")
with open("chain_rule_graph.dot", "w") as ff:
    ff.write(f.to_dot())
print("DOT file saved as chain_rule_graph.dot")
print("To render: dot -Tpng chain_rule_graph.dot -o chain_rule_graph.png")

print("\n" + "=" * 60)
print("CHAIN RULE DEMONSTRATIONS COMPLETE")
print("=" * 60)
