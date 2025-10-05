import numpy as np
from typing import Optional, List, Callable


class Tensor:
    def __init__(self, data, requires_grad=False, grad_fn=None):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = None
        self.grad_fn = grad_fn
        self._backward_hooks: List[Callable] = []

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def backward(self, gradient=None):
        if gradient is None:
            gradient = np.ones_like(self.data)

        if self.requires_grad:
            self.grad = gradient if self.grad is None else self.grad + gradient

        if self.grad_fn is not None:
            self.grad_fn(gradient)

    def _binary_op(self, other, op_fn, grad_fn_factory):
        """Helper for binary operations."""
        is_tensor_other = isinstance(other, Tensor)
        result_data = op_fn(self.data, other.data if is_tensor_other else other)
        requires_grad = self.requires_grad or (is_tensor_other and other.requires_grad)

        result = Tensor(result_data, requires_grad=requires_grad)

        if requires_grad:
            result.grad_fn = grad_fn_factory(self, other, is_tensor_other)

        return result

    def _unary_op(self, op_fn, grad_fn_factory):
        """Helper for unary operations."""
        result_data = op_fn(self.data)
        result = Tensor(result_data, requires_grad=self.requires_grad)

        if self.requires_grad:
            result.grad_fn = grad_fn_factory(self, result_data)

        return result

    def __add__(self, other):
        def grad_fn_factory(self_tensor, other_tensor, is_tensor_other):
            def grad_fn(gradient):
                if self_tensor.requires_grad:
                    self_tensor.backward(gradient)
                if is_tensor_other and other_tensor.requires_grad:
                    other_tensor.backward(gradient)

            return grad_fn

        return self._binary_op(other, np.add, grad_fn_factory)

    def __sub__(self, other):
        def grad_fn_factory(self_tensor, other_tensor, is_tensor_other):
            def grad_fn(gradient):
                if self_tensor.requires_grad:
                    self_tensor.backward(gradient)
                if is_tensor_other and other_tensor.requires_grad:
                    other_tensor.backward(-gradient)

            return grad_fn

        return self._binary_op(other, np.subtract, grad_fn_factory)

    def __mul__(self, other):
        def grad_fn_factory(self_tensor, other_tensor, is_tensor_other):
            def grad_fn(gradient):
                if self_tensor.requires_grad:
                    other_data = other_tensor.data if is_tensor_other else other_tensor
                    self_tensor.backward(gradient * other_data)
                if is_tensor_other and other_tensor.requires_grad:
                    other_tensor.backward(gradient * self_tensor.data)

            return grad_fn

        return self._binary_op(other, np.multiply, grad_fn_factory)

    def __pow__(self, exponent):
        def grad_fn_factory(self_tensor, result_data):
            def grad_fn(gradient):
                # Handle edge cases for numerical stability
                if exponent == 0:
                    local_grad = np.zeros_like(self_tensor.data)
                else:
                    local_grad = exponent * np.power(self_tensor.data, exponent - 1)
                self_tensor.backward(gradient * local_grad)

            return grad_fn

        return self._unary_op(lambda x: np.power(x, exponent), grad_fn_factory)

    def exp(self):
        """Element-wise exponential function."""

        def grad_fn_factory(self_tensor, result_data):
            def grad_fn(gradient):
                self_tensor.backward(gradient * result_data)

            return grad_fn

        return self._unary_op(np.exp, grad_fn_factory)

    def log(self):
        """Element-wise natural logarithm."""

        def grad_fn_factory(self_tensor, result_data):
            def grad_fn(gradient):
                self_tensor.backward(gradient / self_tensor.data)

            return grad_fn

        return self._unary_op(np.log, grad_fn_factory)

    def sum(self, axis=None):
        """Sum reduction along specified axis."""
        result_data = np.sum(self.data, axis=axis)
        result = Tensor(result_data, requires_grad=self.requires_grad)

        if self.requires_grad:

            def grad_fn(gradient):
                if axis is None:
                    grad_expanded = np.full_like(self.data, gradient)
                else:
                    grad_expanded = np.expand_dims(gradient, axis)
                    grad_expanded = np.broadcast_to(grad_expanded, self.data.shape)
                self.backward(grad_expanded)

            result.grad_fn = grad_fn

        return result

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def __matmul__(self, other):
        """Matrix multiplication operation."""
        if not isinstance(other, Tensor):
            raise TypeError(
                "Matrix multiplication requires both operands to be Tensors"
            )

        result_data = np.matmul(self.data, other.data)
        requires_grad = self.requires_grad or other.requires_grad
        result = Tensor(result_data, requires_grad=requires_grad)

        if requires_grad:

            def grad_fn(gradient):
                if self.requires_grad:
                    # For A @ B, gradient w.r.t A is gradient @ B.T
                    self_grad = np.matmul(gradient, other.data.T)
                    self.backward(self_grad)
                if other.requires_grad:
                    # For A @ B, gradient w.r.t B is A.T @ gradient
                    other_grad = np.matmul(self.data.T, gradient)
                    other.backward(other_grad)

            result.grad_fn = grad_fn

        return result
