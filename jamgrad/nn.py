from .tensor import Tensor
import numpy as np


class Linear:
    def __init__(self, in_features, out_features):
        self.weight = Tensor(
            np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features),
            requires_grad=True,
        )
        self.bias = Tensor(np.zeros(out_features), requires_grad=True)

    def __call__(self, x):
        return x @ self.weight + self.bias

    def parameters(self):
        return [self.weight, self.bias]


def relu(x):
    data = np.maximum(0, x.data)
    result = Tensor(data, requires_grad=x.requires_grad)

    if x.requires_grad:

        def grad_fn(gradient):
            relu_grad = (x.data > 0).astype(np.float32)
            x.backward(gradient * relu_grad)

        result.grad_fn = grad_fn

    return result


def softmax(x):
    exp_x = x.exp()
    sum_exp = exp_x.sum(axis=1)

    # Expand dims for broadcasting
    sum_exp_expanded = Tensor(
        np.expand_dims(sum_exp.data, axis=1),
        requires_grad=sum_exp.requires_grad,
    )

    return exp_x * (1.0 / sum_exp_expanded)


def cross_entropy_loss(predictions, targets):
    log_probs = predictions.log()
    # One-hot encoding for targets
    batch_size = targets.data.shape[0]
    loss_sum = Tensor([0.0], requires_grad=True)

    for i in range(batch_size):
        target_idx = int(targets.data[i])
        loss_sum = loss_sum + log_probs.data[i, target_idx] * Tensor([-1.0])

    return loss_sum * (1.0 / batch_size)
