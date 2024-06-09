import numpy as np
import heatflow
from heatflow_cpp import matmul_cpp, add_cpp, subtract_cpp, divide_cpp
import time

a = np.array([[[1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3]]])
matrix_4d = np.array([[[[1], [2], [3]]], [[[4], [5], [6]]]])
# b = np.array([[[1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3]]])

# x = np.array([[[1, 2, 3], [1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3], [1, 2, 3]]])
# y = np.array([[[1, 2, 3], [1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3], [1, 2, 3]]])

x = np.array([1, 2, 3])
# # y = np.array([1, 2, 3])

# print(heatflow.matmul(heatflow.Tensor([1, 2]), heatflow.Tensor([1, 2])))
# print(np.multiply(a, b))
# result = heatflow.Tensor(x) @ heatflow.Tensor(y)
# print(result)
# # tensor_a = heatflow.Tensor(a)
# # tensor_b = heatflow.Tensor(b)

# start_time = time.time()
# result = heatflow.Tensor(x) @ heatflow.Tensor(y)
# # end_time = time.time()
print(heatflow.squeeze(heatflow.Tensor(matrix_4d), axis=3))
# print(heatflow.flatten(a))
# tensor_a = heatflow.Tensor(a)
# print(tensor_a.squeeze(axis=1))
# print(tensor_a.reshape((3, 2, 2)))
# print(np.eye(2, 5))
# print(f"Total Time Taken To Run is: {end_time - start_time}s")

# print(np.matmul(a, b))