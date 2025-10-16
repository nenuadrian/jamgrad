.. jamgrad documentation master file, created by
   sphinx-quickstart on Tue Oct  7 23:08:16 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

jamgrad documentation
=====================

**jamgrad** is a lightweight automatic differentiation library built from first principles.

Overview
--------

jamgrad implements reverse-mode automatic differentiation (autograd) in Python and NumPy,
forming the foundation for building and training neural networks with gradient descent.

Features
~~~~~~~~

* Computational graph construction and visualization
* Automatic gradient computation via backpropagation
* Support for common operations (add, multiply, power, exp, log, etc.)
* DOT graph generation for visualization

Quick Start
-----------

.. code-block:: python

   from jamgrad import Tensor

   # Create tensors with gradient tracking
   x = Tensor([2.0], requires_grad=True)
   y = x ** 2
   
   # Compute gradients
   y.backward()
   print(x.grad)  # dy/dx = 2x = 4.0

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   autoapi/index

