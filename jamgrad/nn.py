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

    # Set operation info for graph visualization
    result._op_name = "ReLU"
    result._children = [x]

    if x.requires_grad:

        def grad_fn(gradient):
            # Derivative of ReLU: 1 if x > 0, else 0
            relu_grad = (x.data > 0).astype(np.float32)
            x.backward(gradient * relu_grad)

        result.grad_fn = grad_fn

    return result


def softmax(x):
    """
    Softmax activation function.

    Applies softmax function along the last dimension:
    softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j

    Args:
        x (Tensor): Input tensor

    Returns:
        Tensor: Output tensor with softmax applied

    Examples:
        >>> x = Tensor([[1, 2, 3]])
        >>> y = softmax(x)  # Probabilities that sum to 1
    """
    # Subtract max for numerical stability
    x_max = Tensor(np.max(x.data, axis=-1, keepdims=True))
    x_shifted = x - x_max

    exp_vals = x_shifted.exp()
    sum_exp = exp_vals.sum(axis=-1)

    # Create a tensor for broadcasting the denominator
    # Expand dimensions to match exp_vals for proper broadcasting
    sum_exp_expanded = Tensor(
        np.expand_dims(sum_exp.data, axis=-1),
        requires_grad=sum_exp.requires_grad
    )
    
    # Copy the gradient function from sum_exp to maintain the chain
    if sum_exp.requires_grad and sum_exp.grad_fn is not None:
        def sum_exp_expanded_grad_fn(gradient):
            # Remove the expanded dimension and pass to original sum_exp
            grad_squeezed = np.sum(gradient, axis=-1)
            sum_exp.backward(grad_squeezed)
        sum_exp_expanded.grad_fn = sum_exp_expanded_grad_fn

    # Use tensor operations to maintain gradient chain
    # This is equivalent to exp_vals / sum_exp_expanded but maintains gradients
    result = exp_vals * (Tensor([1.0]) / sum_exp_expanded)
    
    # Override the operation name for better visualization
    result._op_name = "Softmax"
    result._children = [x]
    
    return result


def cross_entropy_loss(predictions, targets):
    """
    Cross-entropy loss function.

    Args:
        predictions (Tensor): Predicted probabilities from softmax
        targets (Tensor): One-hot encoded target labels

    Returns:
        Tensor: Cross-entropy loss value
    """
    # Clip predictions to avoid log(0)
    eps = 1e-12
    pred_clipped = Tensor(
        np.clip(predictions.data, eps, 1 - eps),
        requires_grad=predictions.requires_grad,
    )
    
    # Copy gradient function to maintain chain
    if predictions.requires_grad and predictions.grad_fn is not None:
        def pred_clipped_grad_fn(gradient):
            # For clipped values, gradient flows through unchanged for values in (eps, 1-eps)
            mask = (predictions.data >= eps) & (predictions.data <= 1-eps)
            grad_filtered = gradient * mask.astype(np.float32)
            predictions.backward(grad_filtered)
        pred_clipped.grad_fn = pred_clipped_grad_fn

    # Compute cross entropy: -sum(targets * log(predictions))
    log_probs = pred_clipped.log()
    loss = -(targets * log_probs).sum() * (1.0 / targets.data.shape[0])
    
    # Override the operation name for better visualization
    loss._op_name = "CrossEntropy"
    loss._children = [predictions, targets]

    return loss
