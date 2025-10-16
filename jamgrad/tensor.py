import numpy as np


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
        self._op_name = None  # Name of the operation that created this tensor
        self._children = []  # Child tensors in computation graph
        self._label = None  # User-defined label for this tensor

    def set_label(self, label):
        """
        Set a human-readable label for this tensor.

        Args:
            label (str): Label for the tensor (e.g., 'x', 'weight', 'loss')

        Returns:
            Tensor: Self for method chaining
        """
        self._label = label
        return self

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

        # Set operation info for graph visualization
        result._op_name = op_fn.__name__.replace("_", "")
        result._children = [self]
        if is_tensor_other:
            result._children.append(other)

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

        # Set operation info for graph visualization
        result._op_name = op_fn.__name__
        result._children = [self]

        if self.requires_grad:
            result.grad_fn = grad_fn_factory(self, result_data)

        return result

    @staticmethod
    def _reduce_broadcast(grad, target_shape):
        """
        Reduce gradient `grad` to match `target_shape` by summing over broadcasted axes.
        This ensures backward() receives a gradient of the same shape as the original tensor.
        """
        # Remove extra leading dimensions that arose from broadcasting
        while grad.ndim > len(target_shape):
            grad = np.sum(grad, axis=0)
        # Sum along axes where target shape had dimension 1
        for i, (gdim, tdim) in enumerate(zip(grad.shape, target_shape)):
            if tdim == 1 and gdim > 1:
                grad = np.sum(grad, axis=i, keepdims=True)
        return grad

    def __add__(self, other):
        """
        Element-wise addition.

        Args:
            other: Tensor or scalar to add

        Returns:
            Tensor: Result of addition with gradient support
        """

        def grad_fn_factory(self_tensor, other_tensor, is_tensor_other):
            def grad_fn(gradient):
                if self_tensor.requires_grad:
                    grad_self = Tensor._reduce_broadcast(gradient, self_tensor.data.shape)
                    self_tensor.backward(grad_self)

                if is_tensor_other and other_tensor.requires_grad:
                    grad_other = Tensor._reduce_broadcast(gradient, other_tensor.data.shape)
                    other_tensor.backward(grad_other)

            return grad_fn

        return self._binary_op(other, np.add, grad_fn_factory)

    def __radd__(self, other):
        """Reverse addition: other + self."""
        return self.__add__(other)

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

    def __rsub__(self, other):
        """Reverse subtraction: other - self."""
        return (-self).__add__(other)

    def __mul__(self, other):
        """
        Element-wise multiplication.

        Args:
            other: Tensor or scalar to multiply

        Returns:
            Tensor: Result of multiplication with gradient support
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

    def __rmul__(self, other):
        """Reverse multiplication: other * self."""
        return self.__mul__(other)

    def __truediv__(self, other):
        """
        Element-wise division.

        Args:
            other: Tensor or scalar to divide by

        Returns:
            Tensor: Result of division with gradient support
        """

        def grad_fn_factory(self_tensor, other_tensor, is_tensor_other):
            def grad_fn(gradient):
                if self_tensor.requires_grad:
                    other_data = other_tensor.data if is_tensor_other else other_tensor
                    self_tensor.backward(gradient / other_data)
                if is_tensor_other and other_tensor.requires_grad:
                    # d/dy (x/y) = -x/y^2
                    other_tensor.backward(
                        -gradient * self_tensor.data / (other_tensor.data**2)
                    )

            return grad_fn

        return self._binary_op(other, np.divide, grad_fn_factory)

    def __rtruediv__(self, other):
        """Reverse division: other / self."""
        if not isinstance(other, Tensor):
            other = Tensor([other] if np.isscalar(other) else other)
        return other.__truediv__(self)

    def __pow__(self, exponent):
        """
        Element-wise power operation.

        Args:
            exponent (float): Power to raise tensor elements to

        Returns:
            Tensor: Result of power operation with gradient support
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
        """
        result_data = np.sum(self.data, axis=axis)
        result = Tensor(result_data, requires_grad=self.requires_grad)

        # Set operation info
        result._op_name = f"sum(axis={axis})"
        result._children = [self]

        if self.requires_grad:

            def grad_fn(gradient):
                if axis is None:
                    # For sum over all elements, broadcast gradient to original shape
                    if np.isscalar(gradient):
                        grad_expanded = np.full_like(self.data, gradient)
                    else:
                        grad_expanded = np.full_like(self.data, gradient.item())
                else:
                    # For sum over specific axis, need to restore the summed dimension
                    if isinstance(axis, int):
                        axes = [axis]
                    else:
                        axes = list(axis)

                    grad_expanded = gradient
                    for ax in sorted(axes):
                        grad_expanded = np.expand_dims(grad_expanded, axis=ax)
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
        """
        if not isinstance(other, Tensor):
            raise TypeError(
                "Matrix multiplication requires both operands to be Tensors"
            )

        result_data = np.matmul(self.data, other.data)
        requires_grad = self.requires_grad or other.requires_grad
        result = Tensor(result_data, requires_grad=requires_grad)

        # Set operation info
        result._op_name = "matmul"
        result._children = [self, other]

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

    def __neg__(self):
        """
        Unary negation.

        Returns:
            Tensor: Negated tensor with gradient support
        """

        def grad_fn_factory(self_tensor, _):
            def grad_fn(gradient):
                self_tensor.backward(-gradient)

            return grad_fn

        return self._unary_op(np.negative, grad_fn_factory)

    def to_dot(self):
        """
        Generate a DOT graph representation of the computation graph (left-to-right).

        Returns:
            str: DOT format string representing the computation graph
        """

        visited_tensors, visited_ops = set(), set()
        edges, tensor_nodes, op_nodes = [], [], []
        traverse(self, visited_tensors, visited_ops, edges, tensor_nodes, op_nodes)

        dot = [
            "digraph ComputationGraph {",
            "  rankdir=LR;",
            "  node [fontsize=9, margin=0.1];",
            "",
            *tensor_nodes,
            "",
            *op_nodes,
            "",
            *edges,
            "}",
        ]
        return "\n".join(dot)


def get_tensor_id(tensor):
    return f"tensor_{id(tensor)}"


def get_op_id(tensor):
    return f"op_{id(tensor)}"


def get_tensor_label(tensor):
    """Build readable label for a tensor node."""
    label_parts = []
    if tensor._label:
        label_parts.append(f"{tensor._label}")
    else:
        label_parts.append("unnamed")

    shape_str = "x".join(map(str, tensor.shape))
    label_parts.append(f"shape=({shape_str})")

    val = np.array2string(tensor.data.flatten()[:3], precision=4, separator=",")
    if tensor.data.size > 3:
        val = val[:-1] + ", ...]"
    label_parts.append(f"val={val}")

    if tensor.requires_grad:
        if tensor.grad is not None:
            grad_str = np.array2string(
                tensor.grad.flatten()[:3], precision=4, separator=","
            )
            if tensor.grad.size > 3:
                grad_str = grad_str[:-1] + ", ...]"
            label_parts.append(f"grad={grad_str}")
        else:
            label_parts.append("grad=None")

    return "\\n".join(label_parts)


def traverse(tensor, visited_tensors, visited_ops, edges, tensor_nodes, op_nodes):
    tid = get_tensor_id(tensor)
    if tid in visited_tensors:
        return
    visited_tensors.add(tid)

    # Tensor node styling
    color = "lightblue" if tensor.requires_grad else "lightgray"
    label = get_tensor_label(tensor)
    tensor_nodes.append(
        f'  {tid} [label="{label}", shape="box", style="filled,rounded", fillcolor="{color}"];'
    )

    # If this tensor came from an operation
    if tensor._op_name and tensor._children:
        opid = get_op_id(tensor)
        if opid not in visited_ops:
            visited_ops.add(opid)
            op_nodes.append(
                f'  {opid} [label="{tensor._op_name}", shape="circle", style="filled", fillcolor="orange"];'
            )

            # Operation → output edge
            edges.append(f"  {opid} -> {tid};")

            # Input → operation edges
            for child in tensor._children:
                cid = get_tensor_id(child)
                edges.append(f"  {cid} -> {opid};")
                traverse(
                    child,
                    visited_tensors,
                    visited_ops,
                    edges,
                    tensor_nodes,
                    op_nodes,
                )
