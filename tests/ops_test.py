import math
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from heatflow import Tensor
from heatflow import matmul, divide, add, subtract, mul
import numpy as np

array_1 = np.array([[[1, 2, 3], [1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3], [1, 2, 3]]])
array_2 = np.array([[[1, 2, 3], [1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3], [1, 2, 3]]])


def testing_basics():    
    print(np.matmul(array_1, array_2))
    print("-------------")
    print(matmul(array_1, array_2))

def testing_grad():
    a = Tensor(array_1, requires_grad=True)
    b = Tensor(array_2, requires_grad=True)

    result = matmul(a, b)
    result.backward()

    print(result)

def test_matmul1():

        a = Tensor([[1, 2], [1, 2]], requires_grad=True)
        b = Tensor([[1, 2, 3], [1, 2, 3]], requires_grad=True)

        result = matmul(a, b)
        result.backward(gradient=Tensor.ones_like(result))

        assert result.tolist() == [[3, 6, 9], [3, 6, 9]]
        assert result.grad.tolist() == [[1, 1, 1], [1, 1, 1]]
        assert a.grad.tolist() == [[6, 6], [6, 6]]
        assert b.grad.tolist() == [[2, 2, 2], [4, 4, 4]]

def test_div():
    a = Tensor(0.154, requires_grad=True)
    b = Tensor(1.565, requires_grad=True)
    res = divide(a, b)

    res.backward()

    assert math.isclose(res.data.tolist(), 0.0984, rel_tol=0.01) == True
    assert math.isclose(a.grad.tolist(), 0.6390, rel_tol=0.01) == True
    assert math.isclose(b.grad.tolist(), -0.0629, rel_tol=0.01) == True

test_div()