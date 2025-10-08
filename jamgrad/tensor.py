import numpy as np
from typing import Optional, List, Callable


class Tensor:
    """
    A tensor class with automatic differentiation capabilities.
    
    This class wraps numpy arrays and provides automatic gradient computation
    for backpropagation in neural networks and optimization algorithms.
    
    Args:
        data: Input data as array-like object
        requires_grad (bool): Whether to compute gradients for this tensor
        grad_fn (callable, optional): Function to compute gradients during backprop
        
    Attributes:
        data (np.ndarray): The underlying numpy array data
        requires_grad (bool): Whether gradients are computed for this tensor
        grad (np.ndarray): Accumulated gradients, None until backward() is called
        grad_fn (callable): Gradient function for backpropagation
        
    Examples:
        >>> x = Tensor([1.0, 2.0], requires_grad=True)
        >>> y = x ** 2
        >>> y.backward(np.ones_like(y.data))
        >>> print(x.grad)  # [2.0, 4.0]
    """
    
    def __init__(self, data, requires_grad=False, grad_fn=None):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = None
        self.grad_fn = grad_fn

    def __repr__(self):
        """String representation of the tensor."""
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def backward(self, gradient=None):
        """
        Compute gradients via backpropagation.
        
        Args:
            gradient (np.ndarray, optional): Gradient from upstream computation.
                If None, assumes gradient of ones (for scalar outputs).
                
        Note:
            This method accumulates gradients in the .grad attribute and
            propagates gradients backward through the computation graph.
        """
        if gradient is None:
            gradient = np.ones_like(self.data)

        if self.requires_grad:
            self.grad = gradient if self.grad is None else self.grad + gradient

        if self.grad_fn is not None:
            self.grad_fn(gradient)

    def _binary_op(self, other, op_fn, grad_fn_factory):
        """
        Helper method for binary operations.
        
        Args:
            other: Second operand (Tensor or scalar)
            op_fn: Numpy function to apply (e.g., np.add, np.multiply)
            grad_fn_factory: Function that creates the gradient function
            
        Returns:
            Tensor: Result of the binary operation
        """
        is_tensor_other = isinstance(other, Tensor)
        result_data = op_fn(self.data, other.data if is_tensor_other else other)
        requires_grad = self.requires_grad or (is_tensor_other and other.requires_grad)

        result = Tensor(result_data, requires_grad=requires_grad)

        if requires_grad:
            result.grad_fn = grad_fn_factory(self, other, is_tensor_other)

        return result

    def _unary_op(self, op_fn, grad_fn_factory):
        """
        Helper method for unary operations.
        
        Args:
            op_fn: Numpy function to apply (e.g., np.exp, np.log)
            grad_fn_factory: Function that creates the gradient function
            
        Returns:
            Tensor: Result of the unary operation
        """
        result_data = op_fn(self.data)
        result = Tensor(result_data, requires_grad=self.requires_grad)

        if self.requires_grad:
            result.grad_fn = grad_fn_factory(self, result_data)

        return result

    def __add__(self, other):
        """
        Element-wise addition.
        
        Args:
            other: Tensor or scalar to add
            
        Returns:
            Tensor: Result of addition with gradient support
            
        Examples:
            >>> a = Tensor([1, 2])
            >>> b = Tensor([3, 4])
            >>> c = a + b  # [4, 6]
        """
        def grad_fn_factory(self_tensor, other_tensor, is_tensor_other):
            def grad_fn(gradient):
                if self_tensor.requires_grad:
                    self_tensor.backward(gradient)
                if is_tensor_other and other_tensor.requires_grad:
                    other_tensor.backward(gradient)

            return grad_fn

        return self._binary_op(other, np.add, grad_fn_factory)

    def __sub__(self, other):
        """
        Element-wise subtraction.
        
        Args:
            other: Tensor or scalar to subtract
            
        Returns:
            Tensor: Result of subtraction with gradient support
        """
        def grad_fn_factory(self_tensor, other_tensor, is_tensor_other):
            def grad_fn(gradient):
                if self_tensor.requires_grad:
                    self_tensor.backward(gradient)
                if is_tensor_other and other_tensor.requires_grad:
                    other_tensor.backward(-gradient)

            return grad_fn

        return self._binary_op(other, np.subtract, grad_fn_factory)

    def __mul__(self, other):
        """
        Element-wise multiplication.
        
        Args:
            other: Tensor or scalar to multiply
            
        Returns:
            Tensor: Result of multiplication with gradient support
            
        Examples:
            >>> a = Tensor([2, 3])
            >>> b = a * 2  # [4, 6]
        """
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
        """
        Element-wise power operation.
        
        Args:
            exponent (float): Power to raise tensor elements to
            
        Returns:
            Tensor: Result of power operation with gradient support
            
        Examples:
            >>> x = Tensor([2, 3])
            >>> y = x ** 2  # [4, 9]
        """
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
        """
        Element-wise exponential function.
        
        Returns:
            Tensor: e^x for each element x in the tensor
            
        Examples:
            >>> x = Tensor([0, 1])
            >>> y = x.exp()  # [1.0, 2.718...]
        """
        def grad_fn_factory(self_tensor, result_data):
            def grad_fn(gradient):
                self_tensor.backward(gradient * result_data)

            return grad_fn

        return self._unary_op(np.exp, grad_fn_factory)

    def log(self):
        """
        Element-wise natural logarithm.
        
        Returns:
            Tensor: ln(x) for each element x in the tensor
            
        Examples:
            >>> x = Tensor([1, 2.718])
            >>> y = x.log()  # [0.0, 1.0]
        """
        def grad_fn_factory(self_tensor, result_data):
            def grad_fn(gradient):
                self_tensor.backward(gradient / self_tensor.data)

            return grad_fn

        return self._unary_op(np.log, grad_fn_factory)

    def sum(self, axis=None):
        """
        Sum reduction along specified axis.
        
        Args:
            axis (int or tuple, optional): Axis or axes along which to sum.
                If None, sum all elements.
                
        Returns:
            Tensor: Sum of tensor elements with gradient support
            
        Examples:
            >>> x = Tensor([[1, 2], [3, 4]])
            >>> y = x.sum()  # 10
            >>> z = x.sum(axis=0)  # [4, 6]
        """
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
        """Shape of the tensor data."""
        return self.data.shape

    @property
    def ndim(self):
        """Number of dimensions of the tensor data."""
        return self.data.ndim

    def __matmul__(self, other):
        """
        Matrix multiplication operation.
        
        Args:
            other (Tensor): Right operand for matrix multiplication
            
        Returns:
            Tensor: Result of matrix multiplication with gradient support
            
        Raises:
            TypeError: If other is not a Tensor
            
        Examples:
            >>> a = Tensor([[1, 2], [3, 4]])
            >>> b = Tensor([[5, 6], [7, 8]])
            >>> c = a @ b  # Matrix multiplication
        """
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
