import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from jamgrad.tensor import Tensor
from jamgrad.nn import Linear, relu, softmax, cross_entropy_loss


class TestBasicGradients:
    """Test basic gradient computation functionality."""

    def test_simple_power_gradient(self):
        """Test gradient computation for simple power function."""
        x = Tensor([2.0], requires_grad=True)
        y = x ** 2
        y.backward()
        
        assert x.grad is not None
        np.testing.assert_array_almost_equal(x.grad, np.array([4.0]))

    def test_multi_operation_gradient(self):
        """Test gradient computation for multiple operations."""
        x = Tensor([1.0, 2.0], requires_grad=True)
        y = (x * 2 + 1).sum()
        y.backward()
        
        assert x.grad is not None
        np.testing.assert_array_almost_equal(x.grad, np.array([2.0, 2.0]))

    def test_chain_rule_gradient(self):
        """Test chain rule implementation."""
        x = Tensor([3.0], requires_grad=True)
        y = x ** 2
        z = y * 2 + 1
        z.backward()
        
        # dz/dx = d/dx(2x^2 + 1) = 4x, at x=3 should be 12
        np.testing.assert_array_almost_equal(x.grad, np.array([12.0]))


class TestLinearLayerGradients:
    """Test linear layer gradient computation."""

    def test_linear_layer_gradients(self):
        """Test that linear layer computes gradients correctly."""
        layer = Linear(2, 1)
        x = Tensor([[1.0, 2.0]], requires_grad=True)
        
        output = layer(x)
        loss = (output ** 2).sum()
        
        # Zero gradients
        layer.weight.grad = None
        layer.bias.grad = None
        x.grad = None
        
        loss.backward()
        
        # Check that all gradients are computed and non-zero
        assert x.grad is not None and np.any(x.grad != 0)
        assert layer.weight.grad is not None and np.any(layer.weight.grad != 0)
        assert layer.bias.grad is not None and np.any(layer.bias.grad != 0)

    def test_linear_vs_torch(self):
        """Compare linear layer gradients with PyTorch."""
        np.random.seed(42)
        torch.manual_seed(42)
        
        # JamGrad
        layer_jam = Linear(3, 2)
        x_jam = Tensor([[1.0, 2.0, -1.0]], requires_grad=True)
        
        # PyTorch with same weights
        layer_torch = nn.Linear(3, 2)
        layer_torch.weight.data = torch.tensor(layer_jam.weight.data.T)
        layer_torch.bias.data = torch.tensor(layer_jam.bias.data)
        
        x_torch = torch.tensor([[1.0, 2.0, -1.0]], requires_grad=True)
        
        # Forward pass
        out_jam = layer_jam(x_jam)
        out_torch = layer_torch(x_torch)
        
        # Loss
        loss_jam = (out_jam ** 2).sum()
        loss_torch = (out_torch ** 2).sum()
        
        # Backward pass
        loss_jam.backward()
        loss_torch.backward()
        
        # Compare gradients
        np.testing.assert_array_almost_equal(
            x_jam.grad, x_torch.grad.numpy(), decimal=4
        )


class TestSoftmaxCrossEntropyGradients:
    """Test softmax and cross-entropy gradient computation."""

    def test_softmax_cross_entropy_gradients(self):
        """Test softmax + cross-entropy gradient computation."""
        logits = Tensor([[1.0, 2.0, 0.5]], requires_grad=True)
        targets = Tensor([[0.0, 1.0, 0.0]])  # One-hot: class 1
        
        # Forward pass
        probs = softmax(logits)
        loss = cross_entropy_loss(probs, targets)
        
        # Backward pass
        logits.grad = None
        loss.backward()
        
        # Check gradients exist and are non-zero
        assert logits.grad is not None and np.any(logits.grad != 0)
        
        # For cross-entropy with softmax, gradient should be (softmax - targets)
        expected_grad = probs.data - targets.data
        np.testing.assert_array_almost_equal(logits.grad, expected_grad, decimal=3)

    def test_softmax_cross_entropy_vs_torch(self):
        """Compare softmax + cross-entropy gradients with PyTorch."""
        # JamGrad
        logits_jam = Tensor([[2.0, 1.0, 0.1]], requires_grad=True)
        targets_jam = Tensor([[1.0, 0.0, 0.0]])
        
        probs_jam = softmax(logits_jam)
        loss_jam = cross_entropy_loss(probs_jam, targets_jam)
        loss_jam.backward()
        
        # PyTorch
        logits_torch = torch.tensor([[2.0, 1.0, 0.1]], requires_grad=True)
        targets_torch = torch.tensor([0], dtype=torch.long)  # Class index for PyTorch
        
        loss_torch = F.cross_entropy(logits_torch, targets_torch)
        loss_torch.backward()
        
        # Compare gradients (should be probs - one_hot_targets)
        np.testing.assert_array_almost_equal(
            logits_jam.grad, logits_torch.grad.numpy(), decimal=4
        )


class TestFullNetworkGradients:
    """Test full network gradient computation."""

    def test_network_learning(self):
        """Test that a simple network can learn (loss decreases)."""
        # Create simple network
        layer1 = Linear(2, 3)
        layer2 = Linear(3, 2)
        
        # Simple training data
        X = Tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=False)
        y = Tensor([[1.0, 0.0], [0.0, 1.0]])  # One-hot encoded
        
        def train_step(lr=0.1):
            # Forward pass
            h1 = relu(layer1(X))
            logits = layer2(h1)
            probs = softmax(logits)
            loss = cross_entropy_loss(probs, y)
            
            # Zero gradients
            for param in layer1.parameters() + layer2.parameters():
                param.grad = None
            
            # Backward pass
            loss.backward()
            
            # Check all parameters have gradients
            for param in layer1.parameters() + layer2.parameters():
                assert param.grad is not None, "Parameter missing gradient"
            
            # Update parameters
            for param in layer1.parameters() + layer2.parameters():
                param.data = param.data - lr * param.grad
            
            return loss.data
        
        # Train for several steps
        losses = []
        for step in range(10):
            loss = train_step()
            losses.append(float(loss) if hasattr(loss, 'item') else loss)
        
        # Check that loss decreased significantly
        assert losses[-1] < losses[0] * 0.8, f"Loss didn't decrease enough: {losses[0]:.4f} -> {losses[-1]:.4f}"

    def test_gradient_flow_through_relu(self):
        """Test that gradients flow properly through ReLU."""
        x = Tensor([[-1.0, 0.0, 1.0, 2.0]], requires_grad=True)
        y = relu(x)
        loss = y.sum()
        
        loss.backward()
        
        # Gradient should be 0 for negative inputs, 1 for positive
        expected_grad = np.array([[0.0, 0.0, 1.0, 1.0]])
        np.testing.assert_array_equal(x.grad, expected_grad)

    def test_broadcasting_gradients(self):
        """Test gradient computation with broadcasting."""
        # Test matrix + vector addition gradients
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = Tensor([0.1, 0.2], requires_grad=True)
        
        y = x + b  # Broadcasting
        loss = y.sum()
        
        loss.backward()
        
        # x gradient should be ones
        np.testing.assert_array_equal(x.grad, np.ones_like(x.data))
        
        # b gradient should be sum along batch dimension
        expected_b_grad = np.array([2.0, 2.0])  # Sum over 2 samples
        np.testing.assert_array_equal(b.grad, expected_b_grad)


if __name__ == "__main__":
    pytest.main([__file__])
