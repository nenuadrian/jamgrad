jamgrad.tensor
==============

.. py:module:: jamgrad.tensor


Classes
-------

.. autoapisummary::

   jamgrad.tensor.Tensor


Functions
---------

.. autoapisummary::

   jamgrad.tensor.get_tensor_id
   jamgrad.tensor.get_op_id
   jamgrad.tensor.get_tensor_label
   jamgrad.tensor.traverse


Module Contents
---------------

.. py:class:: Tensor(data, requires_grad=False, grad_fn=None)

   A tensor class with automatic differentiation capabilities.

   This class wraps numpy arrays and provides automatic gradient computation
   for backpropagation in neural networks and optimization algorithms.

   :param data: Input data as array-like object
   :param requires_grad: Whether to compute gradients for this tensor
   :type requires_grad: bool
   :param grad_fn: Function to compute gradients during backprop
   :type grad_fn: callable, optional

   .. attribute:: data

      The underlying numpy array data

      :type: np.ndarray

   .. attribute:: requires_grad

      Whether gradients are computed for this tensor

      :type: bool

   .. attribute:: grad

      Accumulated gradients, None until backward() is called

      :type: np.ndarray

   .. attribute:: grad_fn

      Gradient function for backpropagation

      :type: callable

   .. rubric:: Examples

   >>> x = Tensor([1.0, 2.0], requires_grad=True)
   >>> y = x ** 2
   >>> y.backward(np.ones_like(y.data))
   >>> print(x.grad)  # [2.0, 4.0]


   .. py:attribute:: data


   .. py:attribute:: requires_grad
      :value: False



   .. py:attribute:: grad
      :value: None



   .. py:attribute:: grad_fn
      :value: None



   .. py:method:: set_label(label)

      Set a human-readable label for this tensor.

      :param label: Label for the tensor (e.g., 'x', 'weight', 'loss')
      :type label: str

      :returns: Self for method chaining
      :rtype: Tensor



   .. py:method:: backward(gradient=None)

      Compute gradients via backpropagation.

      :param gradient: Gradient from upstream computation.
                       If None, assumes gradient of ones (for scalar outputs).
      :type gradient: np.ndarray, optional

      .. note::

         This method accumulates gradients in the .grad attribute and
         propagates gradients backward through the computation graph.



   .. py:method:: exp()

      Element-wise exponential function.

      :returns: e^x for each element x in the tensor
      :rtype: Tensor



   .. py:method:: log()

      Element-wise natural logarithm.

      :returns: ln(x) for each element x in the tensor
      :rtype: Tensor



   .. py:method:: sum(axis=None)

      Sum reduction along specified axis.

      :param axis: Axis or axes along which to sum.
                   If None, sum all elements.
      :type axis: int or tuple, optional

      :returns: Sum of tensor elements with gradient support
      :rtype: Tensor



   .. py:property:: shape


   .. py:property:: ndim

      Number of dimensions of the tensor data.


   .. py:method:: to_dot()

      Generate a DOT graph representation of the computation graph (left-to-right).

      :returns: DOT format string representing the computation graph
      :rtype: str



.. py:function:: get_tensor_id(tensor)

.. py:function:: get_op_id(tensor)

.. py:function:: get_tensor_label(tensor)

   Build readable label for a tensor node.


.. py:function:: traverse(tensor, visited_tensors, visited_ops, edges, tensor_nodes, op_nodes)

