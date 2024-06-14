import inspect
from abc import ABC, abstractmethod
from typing import Dict
import numpy as np
import heatflow

from heatflow import Tensor
import heatflow.nn as nn

# Testing the Linear Model
input_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
input_tensor = Tensor(input_data, requires_grad=True)

linear_layer = nn.Linear(input_dim=3, output_dim=2)
output = linear_layer(input_tensor)

# print("Output:", output)

# Check gradients
# print("Input gradient:", input_tensor.grad)
# print("Weight gradient:", linear_layer.w.grad)
# print("Bias gradient:", linear_layer.b.grad if linear_layer.b.requires_grad else None)


for param, weight in linear_layer.parameters().items():
    print("Param: ", param)
    print("Weight:", weight)

# Zero the gradients
linear_layer.zero_grad()
input_tensor.zero_grad()
