import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from jamgrad.tensor import Tensor
from jamgrad.nn import Linear, relu


class TestLinearLayer:

    def test_linear_forward(self):
        np.random.seed(42)

        linear = Linear(3, 2)
        x = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=False)
        output = linear(x)

        assert output.shape == (2, 2)
        assert output.data.dtype == np.float32

    def test_linear_vs_torch(self):
        np.random.seed(42)

        linear_jam = Linear(3, 2)
        x_jam = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)

        linear_torch = nn.Linear(3, 2)
        linear_torch.weight.data = torch.tensor(linear_jam.weight.data.T)
        linear_torch.bias.data = torch.tensor(linear_jam.bias.data)

        x_torch = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)

        out_jam = linear_jam(x_jam)
        out_torch = linear_torch(x_torch)

        np.testing.assert_array_almost_equal(
            out_jam.data, out_torch.detach().numpy(), decimal=5
        )

    def test_linear_backward_vs_torch(self):
        np.random.seed(42)

        linear_jam = Linear(2, 1)
        x_jam = Tensor([[1.0, 2.0]], requires_grad=True)

        linear_torch = nn.Linear(2, 1)
        linear_torch.weight.data = torch.tensor(linear_jam.weight.data.T)
        linear_torch.bias.data = torch.tensor(linear_jam.bias.data)

        x_torch = torch.tensor([[1.0, 2.0]], requires_grad=True)

        out_jam = linear_jam(x_jam)
        out_torch = linear_torch(x_torch)

        out_jam.backward()
        out_torch.backward()

        np.testing.assert_array_almost_equal(
            x_jam.grad, x_torch.grad.numpy(), decimal=4
        )
        np.testing.assert_array_almost_equal(
            linear_jam.weight.grad, linear_torch.weight.grad.numpy().T, decimal=4
        )


class TestActivationFunctions:

    def test_relu_forward(self):
        x = Tensor([[-1, 0, 1, 2]], requires_grad=True)
        output = relu(x)

        expected = np.array([[0, 0, 1, 2]], dtype=np.float32)
        np.testing.assert_array_equal(output.data, expected)

    def test_relu_vs_torch(self):
        x_jam = Tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]], requires_grad=True)
        x_torch = torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]], requires_grad=True)

        out_jam = relu(x_jam)
        out_torch = F.relu(x_torch)

        np.testing.assert_array_equal(out_jam.data, out_torch.detach().numpy())

        out_jam.backward(np.ones_like(out_jam.data))
        out_torch.backward(torch.ones_like(out_torch))

        np.testing.assert_array_equal(x_jam.grad, x_torch.grad.numpy())


class TestSimpleNetwork:

    def setup_method(self):
        class SimpleNN:
            def __init__(self, input_size, hidden_size, output_size):
                self.layer1 = Linear(input_size, hidden_size)
                self.layer2 = Linear(hidden_size, output_size)

            def __call__(self, x):
                return self.layer2(relu(self.layer1(x)))

            def parameters(self):
                return self.layer1.parameters() + self.layer2.parameters()

        self.SimpleNN = SimpleNN

    def test_network_forward(self):
        np.random.seed(42)
        model = self.SimpleNN(2, 4, 1)
        x = Tensor([[1.0, 2.0]], requires_grad=False)

        output = model(x)
        assert output.shape == (1, 1)

    def test_network_vs_torch_simple(self):
        np.random.seed(42)

        model_jam = self.SimpleNN(2, 3, 1)

        class TorchNet(nn.Module):
            def __init__(self, jam_model):
                super().__init__()
                self.layer1 = nn.Linear(2, 3)
                self.layer2 = nn.Linear(3, 1)

                self.layer1.weight.data = torch.tensor(jam_model.layer1.weight.data.T)
                self.layer1.bias.data = torch.tensor(jam_model.layer1.bias.data)
                self.layer2.weight.data = torch.tensor(jam_model.layer2.weight.data.T)
                self.layer2.bias.data = torch.tensor(jam_model.layer2.bias.data)

            def forward(self, x):
                x = F.relu(self.layer1(x))
                x = self.layer2(x)
                return x

        model_torch = TorchNet(model_jam)

        x_jam = Tensor([[1.0, 2.0]], requires_grad=True)
        x_torch = torch.tensor([[1.0, 2.0]], requires_grad=True)

        out_jam = model_jam(x_jam)
        out_torch = model_torch(x_torch)

        np.testing.assert_array_almost_equal(
            out_jam.data, out_torch.detach().numpy(), decimal=5
        )


if __name__ == "__main__":
    pytest.main([__file__])
