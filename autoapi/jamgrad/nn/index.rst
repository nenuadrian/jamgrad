jamgrad.nn
==========

.. py:module:: jamgrad.nn


Classes
-------

.. autoapisummary::

   jamgrad.nn.Linear


Functions
---------

.. autoapisummary::

   jamgrad.nn.relu
   jamgrad.nn.softmax
   jamgrad.nn.cross_entropy_loss


Module Contents
---------------

.. py:class:: Linear(in_features, out_features)

   Linear (fully connected) layer.

   Applies a linear transformation: y = xW + b

   :param in_features: Size of input features
   :type in_features: int
   :param out_features: Size of output features
   :type out_features: int

   .. attribute:: weight

      Weight matrix of shape (in_features, out_features)

      :type: Tensor

   .. attribute:: bias

      Bias vector of shape (out_features,)

      :type: Tensor

   .. rubric:: Examples

   >>> layer = Linear(784, 128)
   >>> x = Tensor(np.random.randn(32, 784))
   >>> output = layer(x)  # Shape: (32, 128)


   .. py:attribute:: weight


   .. py:attribute:: bias


   .. py:method:: parameters()

      Get all trainable parameters.

      :returns: List containing weight and bias tensors
      :rtype: list



.. py:function:: relu(x)

   Rectified Linear Unit activation function.

   Applies the function element-wise: f(x) = max(0, x)

   :param x: Input tensor
   :type x: Tensor

   :returns: Output tensor with ReLU applied
   :rtype: Tensor

   .. rubric:: Examples

   >>> x = Tensor([-1, 0, 1, 2])
   >>> y = relu(x)  # [0, 0, 1, 2]


.. py:function:: softmax(x)

   Softmax activation function.

   Applies softmax function along the last dimension:
   softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j

   :param x: Input tensor
   :type x: Tensor

   :returns: Output tensor with softmax applied
   :rtype: Tensor

   .. rubric:: Examples

   >>> x = Tensor([[1, 2, 3]])
   >>> y = softmax(x)  # Probabilities that sum to 1


.. py:function:: cross_entropy_loss(predictions, targets)

   Cross-entropy loss function.

   :param predictions: Predicted probabilities from softmax
   :type predictions: Tensor
   :param targets: One-hot encoded target labels
   :type targets: Tensor

   :returns: Cross-entropy loss value
   :rtype: Tensor


