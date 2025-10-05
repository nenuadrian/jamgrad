import pytest
import numpy as np
import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jamgrad.tensor import Tensor


class TestBasicFunctionality:
    """Test basic functionality of the Tensor class."""

    def test_tensor_creation(self):
        """Test tensor creation and basic properties."""
        t = Tensor([1, 2, 3])
        assert t.shape == (3,)
        assert t.ndim == 1
        np.testing.assert_array_equal(t.data, np.array([1, 2, 3], dtype=np.float32))
        assert not t.requires_grad

    def test_tensor_with_grad(self):
        """Test tensor creation with gradient requirement."""
        t = Tensor([1, 2, 3], requires_grad=True)
        assert t.requires_grad
        assert t.grad is None

    def test_tensor_addition(self):
        """Test tensor addition operation."""
        t1 = Tensor([1, 2, 3])
        t2 = Tensor([4, 5, 6])
        result = t1 + t2
        expected = np.array([5, 7, 9], dtype=np.float32)
        np.testing.assert_array_equal(result.data, expected)

    def test_tensor_scalar_addition(self):
        """Test tensor addition with scalar."""
        t = Tensor([1, 2, 3])
        result = t + 5
        expected = np.array([6, 7, 8], dtype=np.float32)
        np.testing.assert_array_equal(result.data, expected)

    def test_tensor_multiplication(self):
        """Test tensor multiplication operation."""
        t1 = Tensor([1, 2, 3])
        t2 = Tensor([2, 3, 4])
        result = t1 * t2
        expected = np.array([2, 6, 12], dtype=np.float32)
        np.testing.assert_array_equal(result.data, expected)


class TestGradientComputation:
    """Test gradient computation functionality."""

    def test_simple_gradient(self):
        """Test computation of simple gradients."""
        x = Tensor([2.0], requires_grad=True)
        y = x**2  # y = x^2
        y.backward()

        # dy/dx = 2x, at x=2 should be 4
        assert x.grad is not None
        np.testing.assert_array_almost_equal(x.grad, np.array([4.0]))

    def test_addition_gradient(self):
        """Test gradients for addition operation."""
        x = Tensor([1.0, 2.0], requires_grad=True)
        y = Tensor([3.0, 4.0], requires_grad=True)
        z = x + y
        z.backward(np.array([1.0, 1.0]))

        # Gradient of addition is 1 for both inputs
        np.testing.assert_array_equal(x.grad, np.array([1.0, 1.0]))
        np.testing.assert_array_equal(y.grad, np.array([1.0, 1.0]))

    def test_multiplication_gradient(self):
        """Test gradients for multiplication operation."""
        x = Tensor([2.0], requires_grad=True)
        y = Tensor([3.0], requires_grad=True)
        z = x * y
        z.backward()

        # dz/dx = y, dz/dy = x
        np.testing.assert_array_equal(x.grad, np.array([3.0]))
        np.testing.assert_array_equal(y.grad, np.array([2.0]))

    def test_chain_rule(self):
        """Test chain rule implementation."""
        x = Tensor([2.0], requires_grad=True)
        y = x**2  # y = x^2
        z = y * 3  # z = 3 * x^2
        z.backward()

        # dz/dx = d/dx(3 * x^2) = 6x, at x=2 should be 12
        np.testing.assert_array_almost_equal(x.grad, np.array([12.0]))

    def test_exp_gradient(self):
        """Test exponential function gradient."""
        x = Tensor([1.0], requires_grad=True)
        y = x.exp()
        y.backward()

        # d/dx exp(x) = exp(x), at x=1 should be e ≈ 2.718
        expected_grad = np.exp(1.0)
        np.testing.assert_array_almost_equal(x.grad, np.array([expected_grad]))

    def test_log_gradient(self):
        """Test logarithm function gradient."""
        x = Tensor([2.0], requires_grad=True)
        y = x.log()
        y.backward()

        # d/dx log(x) = 1/x, at x=2 should be 0.5
        np.testing.assert_array_almost_equal(x.grad, np.array([0.5]))


@pytest.fixture
def sample_tensors():
    """Fixture providing sample tensors for tests."""
    return {
        "x": Tensor([1.0, 2.0, 3.0]),
        "y": Tensor([2.0, 4.0, 6.0]),
        "x_grad": Tensor([1.0, 2.0, 3.0], requires_grad=True),
        "y_grad": Tensor([2.0, 4.0, 6.0], requires_grad=True),
    }


def test_sum_operation(sample_tensors):
    """Test sum operation using fixture."""
    x = sample_tensors["x_grad"]
    result = x.sum()
    result.backward()

    # Gradient of sum is 1 for all elements
    expected_grad = np.ones_like(x.data)
    np.testing.assert_array_equal(x.grad, expected_grad)


@pytest.mark.parametrize(
    "input_val,exponent,expected_grad",
    [
        (2.0, 2, 4.0),  # d/dx x^2 at x=2 is 2*2 = 4
        (3.0, 3, 27.0),  # d/dx x^3 at x=3 is 3*3^2 = 27
        (2.0, 0.5, 0.35355339),  # d/dx x^0.5 at x=2 is 0.5/sqrt(2) ≈ 0.35355
    ],
)
def test_power_gradients(input_val, exponent, expected_grad):
    """Parametrized test for power function gradients."""
    x = Tensor([input_val], requires_grad=True)
    y = x**exponent
    y.backward()

    np.testing.assert_array_almost_equal(x.grad, np.array([expected_grad]), decimal=5)


class TestTorchComparison:
    """Test gradient computation against PyTorch for validation."""

    def test_simple_gradient_vs_torch(self):
        """Compare simple gradient computation with PyTorch."""
        # JamGrad
        x_jam = Tensor([2.0], requires_grad=True)
        y_jam = x_jam**2
        y_jam.backward()

        # PyTorch
        x_torch = torch.tensor([2.0], requires_grad=True)
        y_torch = x_torch**2
        y_torch.backward()

        np.testing.assert_array_almost_equal(
            x_jam.grad, x_torch.grad.numpy(), decimal=5
        )

    def test_multiplication_gradient_vs_torch(self):
        """Compare multiplication gradients with PyTorch."""
        # JamGrad
        x_jam = Tensor([2.0, 3.0], requires_grad=True)
        y_jam = Tensor([4.0, 5.0], requires_grad=True)
        z_jam = x_jam * y_jam
        z_jam.backward(np.array([1.0, 1.0]))

        # PyTorch
        x_torch = torch.tensor([2.0, 3.0], requires_grad=True)
        y_torch = torch.tensor([4.0, 5.0], requires_grad=True)
        z_torch = x_torch * y_torch
        z_torch.backward(torch.tensor([1.0, 1.0]))

        np.testing.assert_array_almost_equal(
            x_jam.grad, x_torch.grad.numpy(), decimal=5
        )
        np.testing.assert_array_almost_equal(
            y_jam.grad, y_torch.grad.numpy(), decimal=5
        )

    def test_chain_rule_vs_torch(self):
        """Compare chain rule implementation with PyTorch."""
        # JamGrad
        x_jam = Tensor([2.0], requires_grad=True)
        y_jam = x_jam**2
        z_jam = y_jam * 3
        z_jam.backward()

        # PyTorch
        x_torch = torch.tensor([2.0], requires_grad=True)
        y_torch = x_torch**2
        z_torch = y_torch * 3
        z_torch.backward()

        np.testing.assert_array_almost_equal(
            x_jam.grad, x_torch.grad.numpy(), decimal=5
        )

    def test_exp_gradient_vs_torch(self):
        """Compare exponential gradients with PyTorch."""
        # JamGrad
        x_jam = Tensor([1.0, 2.0], requires_grad=True)
        y_jam = x_jam.exp()
        y_jam.backward(np.array([1.0, 1.0]))

        # PyTorch
        x_torch = torch.tensor([1.0, 2.0], requires_grad=True)
        y_torch = torch.exp(x_torch)
        y_torch.backward(torch.tensor([1.0, 1.0]))

        np.testing.assert_array_almost_equal(
            x_jam.grad, x_torch.grad.numpy(), decimal=5
        )

    def test_log_gradient_vs_torch(self):
        """Compare logarithm gradients with PyTorch."""
        # JamGrad
        x_jam = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        y_jam = x_jam.log()
        y_jam.backward(np.array([1.0, 1.0, 1.0]))

        # PyTorch
        x_torch = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y_torch = torch.log(x_torch)
        y_torch.backward(torch.tensor([1.0, 1.0, 1.0]))

        np.testing.assert_array_almost_equal(
            x_jam.grad, x_torch.grad.numpy(), decimal=5
        )

    def test_sum_gradient_vs_torch(self):
        """Compare sum gradients with PyTorch."""
        # JamGrad
        x_jam = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y_jam = x_jam.sum()
        y_jam.backward()

        # PyTorch
        x_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y_torch = torch.sum(x_torch)
        y_torch.backward()

        np.testing.assert_array_almost_equal(
            x_jam.grad, x_torch.grad.numpy(), decimal=5
        )

    @pytest.mark.parametrize("exponent", [2, 3, 0.5, -1])
    def test_power_gradient_vs_torch(self, exponent):
        """Compare power function gradients with PyTorch."""
        # JamGrad
        x_jam = Tensor([2.0, 3.0], requires_grad=True)
        y_jam = x_jam**exponent
        y_jam.backward(np.array([1.0, 1.0]))

        # PyTorch
        x_torch = torch.tensor([2.0, 3.0], requires_grad=True)
        y_torch = x_torch**exponent
        y_torch.backward(torch.tensor([1.0, 1.0]))

        np.testing.assert_array_almost_equal(
            x_jam.grad, x_torch.grad.numpy(), decimal=4
        )

    def test_complex_expression_vs_torch(self):
        """Compare complex expression gradients with PyTorch."""
        # Complex expression: z = (x^2 + y) * exp(x) + log(y)

        # JamGrad
        x_jam = Tensor([1.0], requires_grad=True)
        y_jam = Tensor([2.0], requires_grad=True)
        z_jam = (x_jam**2 + y_jam) * x_jam.exp() + y_jam.log()
        z_jam.backward()

        # PyTorch
        x_torch = torch.tensor([1.0], requires_grad=True)
        y_torch = torch.tensor([2.0], requires_grad=True)
        z_torch = (x_torch**2 + y_torch) * torch.exp(x_torch) + torch.log(y_torch)
        z_torch.backward()

        np.testing.assert_array_almost_equal(
            x_jam.grad, x_torch.grad.numpy(), decimal=4
        )
        np.testing.assert_array_almost_equal(
            y_jam.grad, y_torch.grad.numpy(), decimal=4
        )


if __name__ == "__main__":
    pytest.main([__file__])
