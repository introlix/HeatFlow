import math
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from heatflow import Tensors
from heatflow import matmul, div, add, sub, mul
import numpy as np

array_1 = np.array([[[1, 2, 3], [1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3], [1, 2, 3]]])
array_2 = np.array([[[1, 2, 3], [1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3], [1, 2, 3]]])


def testing_basics():    
    print(np.matmul(array_1, array_2))
    print("-------------")
    print(matmul(array_1, array_2))

def testing_grad():
    a = Tensors(array_1, requires_grad=True)
    b = Tensors(array_2, requires_grad=True)

    result = matmul(a, b)
    result.backward()

    print(result)

def test_matmul1():

        a = Tensors([[1, 2], [1, 2]], requires_grad=True)
        b = Tensors([[1, 2, 3], [1, 2, 3]], requires_grad=True)

        result = a @ b
        result.backward(gradient=Tensors.ones_like(result))

        print(b.grad.tolist())

        assert result.tolist() == [[3, 6, 9], [3, 6, 9]]
        assert a.grad.tolist() == [[6, 6], [6, 6]]
        assert b.grad.tolist() == [[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]]

def test_div():
    a = Tensors(0.154, requires_grad=True)
    b = Tensors(1.565, requires_grad=True)
    res = div(a, b)

    res.backward(gradient=Tensors.ones_like(res))

    assert math.isclose(res.data.tolist(), 0.0984, rel_tol=0.01) == True
    assert math.isclose(a.grad.tolist(), 0.6390, rel_tol=0.01) == True
    assert math.isclose(b.grad.tolist(), -0.0629, rel_tol=0.01) == True

def test_mul():
    a = Tensors([[1, 2, 3], [1, 2, 3]], requires_grad=True)
    b = Tensors([[1, 2, 3], [1, 2, 3]], requires_grad=True)

    result = mul(a, b)
    result.backward(gradient=Tensors.ones_like(result))

    assert result.tolist() == [[1, 4, 9], [1, 4, 9]]
    assert a.grad.tolist() == [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
    assert b.grad.tolist() == [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]

def test_add():
    a = Tensors([[1, 2, 3], [1, 2, 3]], requires_grad=True)
    b = Tensors([[1, 2, 3], [1, 2, 3]], requires_grad=True)

    result = add(a, b)
    result.backward(gradient=Tensors.ones_like(result))

    assert result.tolist() == [[2, 4, 6], [2, 4, 6]]
    assert a.grad.tolist() == [[1, 1, 1], [1, 1, 1]]
    assert b.grad.tolist() == [[1, 1, 1], [1, 1, 1]]

def test_subtract():
    a = Tensors([[1, 2, 3], [1, 2, 3]], requires_grad=True)
    b = Tensors([[1, 2, 3], [1, 2, 3]], requires_grad=True)

    result = sub(a, b)
    result.backward(gradient=Tensors.ones_like(result))

    assert result.tolist() == [[0, 0, 0], [0, 0, 0]]
    assert a.grad.tolist() == [[1, 1, 1], [1, 1, 1]]
    assert b.grad.tolist() == [[-1, -1, -1], [-1, -1, -1]]

test_matmul1()