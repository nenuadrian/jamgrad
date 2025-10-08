from .tensor import Tensor
import numpy as np


class Linear:
    """
    Linear (fully connected) layer.

    Applies a linear transformation: y = xW + b

    Args:
        in_features (int): Size of input features
        out_features (int): Size of output features

    Attributes:
        weight (Tensor): Weight matrix of shape (in_features, out_features)
        bias (Tensor): Bias vector of shape (out_features,)

    Examples:
        >>> layer = Linear(784, 128)
        >>> x = Tensor(np.random.randn(32, 784))
        >>> output = layer(x)  # Shape: (32, 128)
    """

    def __init__(self, in_features, out_features):
        # Xavier/Glorot initialization
        std = np.sqrt(2.0 / in_features)
        self.weight = Tensor(
            np.random.randn(in_features, out_features) * std,
            requires_grad=True,
        )
        self.bias = Tensor(np.zeros(out_features), requires_grad=True)

    def __call__(self, x):
        """
        Forward pass through the linear layer.

        Args:
            x (Tensor): Input tensor of shape (..., in_features)

        Returns:
            Tensor: Output tensor of shape (..., out_features)
        """
        return x @ self.weight + self.bias

    def parameters(self):
        """
        Get all trainable parameters.

        Returns:
            list: List containing weight and bias tensors
        """
        return [self.weight, self.bias]


def relu(x):
    """
    Rectified Linear Unit activation function.

    Applies the function element-wise: f(x) = max(0, x)

    Args:
        x (Tensor): Input tensor

    Returns:
        Tensor: Output tensor with ReLU applied

    Examples:
        >>> x = Tensor([-1, 0, 1, 2])
        >>> y = relu(x)  # [0, 0, 1, 2]
    """
    data = np.maximum(0, x.data)
    result = Tensor(data, requires_grad=x.requires_grad)

    if x.requires_grad:

        def grad_fn(gradient):
            # Derivative of ReLU: 1 if x > 0, else 0
            relu_grad = (x.data > 0).astype(np.float32)
            x.backward(gradient * relu_grad)

        result.grad_fn = grad_fn

    return result


def softmax(x):
    exp_vals = x.exp()
    sum_exp = exp_vals.sum(axis=1)

    # Manual broadcasting for division
    result_data = np.zeros_like(exp_vals.data)
    for i in range(exp_vals.data.shape[0]):
        result_data[i] = exp_vals.data[i] / sum_exp.data[i]

    result = Tensor(result_data, requires_grad=x.requires_grad)

    if x.requires_grad:

        def grad_fn(gradient):
            # Simplified softmax gradient
            s = result.data
            grad_input = np.zeros_like(s)
            for i in range(s.shape[0]):
                jacobian = np.diag(s[i]) - np.outer(s[i], s[i])
                grad_input[i] = gradient[i] @ jacobian
            x.backward(grad_input)

        result.grad_fn = grad_fn

    return result


def cross_entropy_loss(predictions, targets):
    log_probs = predictions.log()
    # One-hot encoding for targets
    batch_size = targets.data.shape[0]
    loss_sum = Tensor([0.0], requires_grad=True)

    for i in range(batch_size):
        target_idx = int(targets.data[i])
        loss_sum = loss_sum + log_probs.data[i, target_idx] * Tensor([-1.0])

    return loss_sum * (1.0 / batch_size)
