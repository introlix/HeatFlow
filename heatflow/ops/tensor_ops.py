from heatflow._tensor import Tensor
import numpy as np

def zeros(*shape) -> Tensor:
    """
    Generate a tensor filled with zeros.

    Args:
        shape (tuple): The shape of the matrix.

    Returns:
        Tensor: A tensor of the given shape filled with zeros.
    """
    return Tensor(np.zeros(shape))


def ones(*shape) -> Tensor:
    """
    Generate a tensor filled with ones.

    Args:
        shape (tuple): The shape of the tensor.

    Returns:
        Tensor: A tensor of the given shape filled with ones.
    """
    return Tensor(np.ones(shape))

def random(*shape) -> Tensor:
    """
    Generate a tensor filled with random values.

    Args:
        shape (tuple): The shape of the tensor.

    Returns:
        Tensor: A tensor of the given shape filled with random values.
    """

    return Tensor(np.random.random(shape))

def eye(m: int, n: int = None) -> Tensor:
    """
    Generate an identity tensor with given shape

    Args:
        shape: shape for the identity tensor

    Returns:
        identiy (Tensor)
    """

    if n == None:
        n = m

    return Tensor(np.eye(m, n))


def sum(tensor, axis=None) -> Tensor:
    """
    Compute the sum of elements in a tensor along a given axis.

    Args:
        tensor (Tensor, ndarray): The input tensor.
        axis (int, optional): The axis along which to sum. 
                          If None, sum all the elements.

    Returns:
        Tensor: The sum of elements along the specified axis.
    """
    if isinstance(tensor, Tensor):
        tensor = tensor.data
    return Tensor(np.sum(tensor, axis=axis))

def mean(tensor, axis=None) -> Tensor:
    """
    Compute the mean of elements in a tensor along a given axis.

    Args:
        tensor (Tensor, ndarray): The input tensor.
        axis (int, optional): The axis along which to calculate mean. 
                          If None, mean all the elements.

    Returns:
        Tensor: The mean of elements along the specified axis.
    """
    if isinstance(tensor, Tensor):
        tensor = tensor.data
    return Tensor(np.mean(tensor, axis=axis))

def max(tensor, axis=None) -> Tensor:
    """
    Compute the max value of elements in a tensor along a given axis.

    Args:
        tensor (Tensor, ndarray): The input tensor.
        axis (int, optional): The axis along which to calculate max value. 

    Returns:
        Tensor: The max value of elements along the specified axis.
    """
    if isinstance(tensor, Tensor):
        tensor = tensor.data
    return Tensor(np.max(tensor, axis=axis))

def min(tensor, axis=None) -> Tensor:
    """
    Compute the min value of elements in a tensor along a given axis.

    Args:
        tensor (Tensor, ndarray): The input tensor.
        axis (int, optional): The axis along which to calculate min value. 

    Returns:
        Tensor: The min value of elements along the specified axis.
    """
    if isinstance(tensor, Tensor):
        tensor = tensor.data
    return Tensor(np.min(tensor, axis=axis))

def prod(tensor, axis=None) -> Tensor:
    """
    Compute the product of elements in a tensor along a given axis.

    Args:
        tensor (Tensor, ndarray): The input tensor.
        axis (int, optional): The axis along which to calculate product. 

    Returns:
        Tensor: The product of elements along the specified axis.
    """
    if isinstance(tensor, Tensor):
        tensor = tensor.data
    return Tensor(np.prod(tensor, axis=axis))


# Shape Manipulation Functions

def reshape(tensor, newshape) -> Tensor:
    """
    Reshape tensor to a new shape.

    Args:
        tensor (Tensor, ndarray): The input tenosr.
    Returns:
        Tensor: A new tensor with new shape
    """
    if isinstance(tensor, Tensor):
        tensor = tensor.data
    return Tensor(np.reshape(tensor, newshape))

def flatten(tensor) -> Tensor:
    """
    Flatten tensor to 1D

    Args:
        tensor (Tensor, ndarray): The input tensor
    Returns:
        Tensor: A new 1D Tensor
    """

    if isinstance(tensor, Tensor):
        tensor = tensor.data
    return Tensor(tensor.flatten())

def expand_dims(tensor, axis) -> Tensor:
    """
    Expand the dimensions of a given tensor by adding a new axis at the specified position.

    Args:
        tensor (tensor, ndarray): The input tensor.
        axis (int): The position at which to insert the new axis.

    Returns:
        Tensor: The matrix with the expanded dimensions.
    """
    if isinstance(tensor, Tensor):
        tensor = tensor.data
    return Tensor(np.expand_dims(tensor, axis))

def squeeze(tensor, axis=None) -> Tensor:
    """
    Remove single-dimensional entries from the shape of a given tensor.

    Args:
        tensor (np.ndarray): The input tensor.
        axis (int or tuple of int, optional): Selects a subset of the single-dimensional entries in the shape. 
                                          If an axis is selected with shape entry greater than one, an error is raised.

    Returns:
        Tensor: The tensor with single-dimensional entries removed.
    """
    if isinstance(tensor, Tensor):
        tensor = tensor.data
    return Tensor(np.squeeze(tensor, axis=axis))