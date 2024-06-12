"""
HeatFlow
========

HeatFlow is a python framework to work with neural networks, tensor, etc. It also provides some useful functions for machine learning and deep learning. HeatFlow gives you full control on your code. It is written on top of numpy which makes it blazing fast compared to other. It uses jit to make the code run as fast as possible.
"""

from . import heatflow_cpp
from heatflow._tensor import Tensor, toTensor
from heatflow.ops.basics import matmul, add, subtract, divide, mul
from heatflow.ops.tensor_ops import zeros, ones, random, eye, sum, mean, max, min, prod, reshape, flatten, expand_dims, squeeze