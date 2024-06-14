import heatflow._tensor
import numpy as np
import heatflow
from heatflow import Tensor
from typing import Tuple
from heatflow_cpp import matmul_cpp, add_cpp, subtract_cpp, divide_cpp
from utils import flatten_nd_array_to_2d_list, reconstruct_nd_array_from_2d_list, registerFn

# Support broadcasting issue in backwards
def manageBroadcasting(
    input_ndim: int, input_shape: Tuple[int], local_gradient: np.ndarray
) -> np.ndarray:
    """Handles broadcasting issue when computing gradients when the output gradient is broadcasted to the inputs.

    Parameters
    ----------
    Arg: input_ndim
        Rank of the tensor for which the gradient is being computed

    Arg: input_shape
        Shape of the tensor for gradient calculation

    Arg: local_gradient
        Gradient that is propogated from the output tensor.

    """

    # Given the gradient of the output is scalar there is no need for broadcasting
    if type(local_gradient) in [np.float32, float] or input_ndim > local_gradient.ndim:
        return local_gradient

    drop_dim: int = local_gradient.ndim - input_ndim
    for _ in range(drop_dim):
        local_gradient = local_gradient.sum(axis=0)

    # What is happening?
    # As we have already normalized the rank, we just sum over the dim while retaining dim
    # (2,3) + (1,3) => (2,3) :
    # (1,3) is broadcasted, so essentially we just have to sum over the _gradient along the dim which is equal to that of the child.

    for i, dim in enumerate(input_shape):
        if dim == 1:
            local_gradient = local_gradient.sum(axis=i, keepdims=True)

    return local_gradient


@registerFn(heatflow.Tensor, "__mul__")
def mul(a, b) -> Tensor:
    """
    Dot product two multiply two matrix.

    Forward:
        Compute the matrix multiplication of two tensors:
            c = a * b

    Backward:
        For gradient computation:
            ∂c/∂a = b
            ∂c/∂b = a
        Thus:
            grad_a = grad_c * b
            grad_b = grad_c * a
    """
    a = heatflow.toTensor(a)
    b = heatflow.toTensor(b)
    output = heatflow.Tensor(np.multiply(a.data, b.data), requires_grad=(a.requires_grad or b.requires_grad))
    output.save_for_backward([a, b])

    def grad_fn():
        if a.requires_grad:
            a_local_gradient = np.multiply(output.grad.data, b.data)
            a_local_gradient = manageBroadcasting(a.ndim, a.shape, a_local_gradient)
            a.grad.data += a_local_gradient
        
        if b.requires_grad:
            b_local_gradient = np.multiply(output.grad.data, a.data)
            b_local_gradient = manageBroadcasting(b.ndim, b.shape, b_local_gradient)
            b.grad.data += b_local_gradient
    
    output.grad_fn = grad_fn

    return output

@registerFn(Tensor, "__matmul__")
def matmul(a, b) -> Tensor:
    """
    Return result of matrix multiplication of the inputs

    Forward:
        Compute the matrix multiplication of two tensors:
            c = a @ b

    Backward:
        For gradient computation:
            ∂c/∂a = b^T
            ∂c/∂b = a^T
        Thus:
            grad_a = grad_c @ b^T
            grad_b = grad_c @ a^T
    """

    a = heatflow.toTensor(a)
    b = heatflow.toTensor(b)

    if a.ndim == 0 or b.ndim == 0:
        raise RuntimeError(
            f"Inputs dimensions to matmul needs to be atleast 1D-Tensor."
        )

    try:
        data = a.data @ b.data
    except ValueError:
        raise TypeError(
            f"Inconsistent tensor size for the operation. {a.shape} x {b.shape} != (m,n) x (n,k)"
        )

    output = heatflow.Tensor(data=data, requires_grad=(a.requires_grad or b.requires_grad))
    output.save_for_backward([a, b])

    def grad_fn():

        if a.requires_grad:

            a_local_gradient = np.dot(output.grad.data, b.data.T)
            a_local_gradient = manageBroadcasting(a.ndim, a.shape, a_local_gradient)

            a.grad.data += a_local_gradient.reshape(a.grad.shape)

        if b.requires_grad:

            b_local_gradient = np.dot(a.data.T, output.grad.data)
            b_local_gradient = manageBroadcasting(b.ndim, b.shape, b_local_gradient)

            b.grad.data += b_local_gradient.reshape(b.grad.shape)

    output.grad_fn = grad_fn

    return output

@registerFn(heatflow.Tensor, "__add__")    
def add(a, b) -> Tensor:
    """
    Adds two n-dimensional numpy arrays using Eigen's matrix addition.

    Forward:
        Compute the element wise addition of two tensor:
            c = a + b
    
    Backward:
        For gradient computation:
            ∂c/∂a = 1
            ∂c/∂b = 1
        Thus:
            grad_a = grad_c * 1
            grad_b = grad_c * 1
    """
    a = heatflow.toTensor(a)
    b = heatflow.toTensor(b)

    output = heatflow.Tensor(a.data + b.data, requires_grad=(a.requires_grad or b.requires_grad))
    output.save_for_backward([a, b])

    def grad_fn():

        if a.requires_grad:

            a_local_gradient = output.grad.data * np.ones_like(a.data)
            a_local_gradient = manageBroadcasting(a.ndim, a.shape, a_local_gradient)

            a.grad.data += a_local_gradient

        if b.requires_grad:

            b_local_gradient = output.grad.data * np.ones_like(b.data)
            b_local_gradient = manageBroadcasting(b.ndim, b.shape, b_local_gradient)

            b.grad.data += b_local_gradient

    output.grad_fn = grad_fn

    return output

@registerFn(heatflow.Tensor, "__sub__")
def subtract(a, b) -> Tensor:
    """
    Subtracts two n-dimensional numpy arrays using Eigen's matrix multiplication.

    Forward:
        Compute the element wise subtraction of two tensor:
            c = a - b
    
    Backward:
        For gradient computation:
            ∂c/∂a = 1
            ∂c/∂b = 1
        Thus:
            grad_a = grad_c * 1
            grad_b = grad_c * -1
    """

    a = heatflow.toTensor(a)
    b = heatflow.toTensor(b)

    output = heatflow.Tensor(a.data - b.data, requires_grad=(a.requires_grad or b.requires_grad))
    output.save_for_backward([a, b])

    def grad_fn():

        if a.requires_grad:

            a_local_gradient = output.grad.data * np.ones_like(a.data)
            a_local_gradient = manageBroadcasting(a.ndim, a.shape, a_local_gradient)

            a.grad.data += a_local_gradient

        if b.requires_grad:

            b_local_gradient = output.grad.data * -1.0 * np.ones_like(b.data)
            b_local_gradient = manageBroadcasting(b.ndim, b.shape, b_local_gradient)

            b.grad.data += b_local_gradient

    output.grad_fn = grad_fn

    return output

@registerFn(heatflow.Tensor, "__truediv__")    
def divide(a, b) -> Tensor:
    """
    Divides two n-dimensional numpy arrays using Eigen's matrix multiplication.

    Forward:
        Compute the element wise subtraction of two tensor:
            c = a / b
    
    Backward:
        For gradient computation:
            ∂c/∂a = 1/b
            ∂c/∂b = -a/b^2
        Thus:
            grad_a = grad_c * 1/b
            grad_b = grad_c * -a/b^2
    """

    a = heatflow.toTensor(a)
    b = heatflow.toTensor(b)
    
    inv_b = b ** -1

    output = heatflow.Tensor(
        a.data * inv_b.data, requires_grad=(a.requires_grad or inv_b.requires_grad)
    )
    output.save_for_backward([a, b, inv_b])


    def grad_fn():

        if a.requires_grad:

            a_local_gradient = output.grad.data * inv_b.data
            a_local_gradient = manageBroadcasting(a.ndim, a.shape, a_local_gradient)

            a.grad.data += a_local_gradient

        if inv_b.requires_grad:

            inv_b_local_gradient = output.grad.data * a.data
            b_local_gradient = -inv_b_local_gradient * (inv_b.data **2)
            b_local_gradient = manageBroadcasting(
                b.ndim, b.shape, b_local_gradient
            )

            b.grad.data += b_local_gradient

    output.grad_fn = grad_fn

    return output

@registerFn(heatflow.Tensor, "__pow__")
def pow(a, pow) -> Tensor:
    """
    Calculate the power of a tensor.
    """

    a = heatflow.toTensor(a)

    output = heatflow.Tensor(a.data ** (pow), requires_grad=a.requires_grad)
    output.save_for_backward([a])

    def grad_fn():

        if a.requires_grad:
            operation_gradient = pow * (a.data ** (pow - 1))
            local_gradient = output.grad.data * operation_gradient
            local_gradient = manageBroadcasting(a.ndim, a.shape, local_gradient)

            a.grad.data += local_gradient

    output.grad_fn = grad_fn
    return output

@registerFn(Tensor, "sum")
def sum(a, axis: int = None):

    a = heatflow.toTensor(a)
    sum_data = a.data.sum() if axis is None else a.data.sum(axis=axis)

    output = Tensor(data=sum_data, requires_grad=a.requires_grad)
    output.save_for_backward([a])

    def backward_fn():

        if a.requires_grad:
            local_gradient = output.grad.data * np.ones_like(a)
            local_gradient = manageBroadcasting(a.ndim, a.shape, local_gradient)
            a.grad.data += local_gradient

    output.backward_fn = backward_fn

    return output

# functions
@registerFn(Tensor, "sigmoid")
def sigmoid(x: Tensor) -> Tensor:
    output_data = 1.0 / (1.0 + np.exp(-x.data))  # sig(x)

    output = Tensor(output_data, requires_grad=x.requires_grad)
    output.save_for_backward([x])

    def grad_fn():

        if x.requires_grad:

            sigmoid_grad = output.data * (1 - output.data)  # sig(x) * (1-sig(x))
            local_gradient = output.grad.data * sigmoid_grad

            x.grad.data += local_gradient

    output.grad_fn = grad_fn

    return output


@registerFn(Tensor, "tanh")
def tanh(x: Tensor) -> Tensor:

    tanh_x = np.tanh(x.data)

    output = Tensor(tanh_x, requires_grad=x.requires_grad)
    output.save_for_backward([x])

    def grad_fn():

        if x.requires_grad:

            local_gradient = output.grad.data * (1 - (tanh_x * tanh_x))
            x.grad.data += local_gradient

    output.grad_fn = grad_fn

    return output


@registerFn(Tensor, "relu")
def relu(x) -> Tensor:

    output = Tensor(np.maximum(x.data, 0), requires_grad=x.requires_grad)
    output.save_for_backward([x])

    def grad_fn():

        if x.requires_grad:

            local_gradient = (x.data >= 0) * output.grad.data
            x.grad.data += local_gradient

    output.grad_fn = grad_fn

    return output


@registerFn(Tensor, "softmax")
def softmax(x: Tensor) -> Tensor:
    """
    Softmax function suffers from numerical error hence must be stabilized against overflow and underflow.

    softmax(x)_i = exp(x)_i / sum(exp(x))

    When x_i is a large negative number, exp(x_i) will underflow and approximate it to zero.
    This results in denominator tending to zero -> nan

    """

    ax = x.ndim - 1
    dim = x.shape[:-1] + (1,)

    x_norm = x.data - x.data.max(axis=ax).reshape(dim)
    x_exp: np.ndarray = np.exp(x_norm)

    output_data = x_exp / x_exp.sum(axis=ax).reshape(dim)

    output = Tensor(output_data, requires_grad=x.requires_grad)
    output.save_for_backward([x])

    def grad_fn():

        if x.requires_grad:

            local_gradient = output.grad.data * output_data * (1.0 - output_data)
            x.grad.data += local_gradient

    output.grad_fn = grad_fn

    return output