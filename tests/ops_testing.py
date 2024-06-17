import heatflow
from heatflow import Tensor

a = Tensor([[[1, 2, 3, 4], [1, 2, 3, 4]], [[1, 2, 3, 4], [1, 2, 3, 4]]])

print(a.shape)
print(a.flatten())
print(a.expand_dims(axis=1))
print(a.squeeze())
print("-------------------------/n")
print(heatflow.flatten(a))
print(heatflow.squeeze(a))
print(heatflow.zeros(2, 2, 5))
print(heatflow.ones(2, 2, 2))
print(heatflow.random(2, 2, 2))
print(heatflow.eye(2, 2))