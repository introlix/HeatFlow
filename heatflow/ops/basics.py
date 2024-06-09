import numpy as np
import heatflow
from numba import njit
from heatflow_cpp import matmul_cpp, add_cpp, subtract_cpp, divide_cpp
from utils import flatten_nd_array_to_2d_list, reconstruct_nd_array_from_2d_list

def mul(array1, array2):
    """
    Dot product two multiply two matrix
    """
    if isinstance(array1, heatflow.Tensor):
        array1 = array1.data
    
    if isinstance(array2, heatflow.Tensor):
        array2 = array2.data

    return heatflow.Tensor(np.multiply(array1, array2))

def matmul(array1, array2):
    """
    Multiplies two n-dimensional numpy arrays using Eigen's matrix multiplication.
    """
    if isinstance(array1, heatflow.Tensor):
        array1 = array1.data
    
    if isinstance(array2, heatflow.Tensor):
        array2 = array2.data

    if array1.shape != array2.shape:
        raise ValueError("The two arrays must have the same shape.")
    

    # Flatten the n-dimensional arrays to lists of 2D arrays
    if array1.ndim > 2:
        if array1.shape[-1] != array2.shape[-2]:
            raise ValueError(f"The tensor with shape {array1.shape} can not be multiplied with tensor with shape {array2.shape}")
        
        list1 = flatten_nd_array_to_2d_list(array1)
        list2 = flatten_nd_array_to_2d_list(array2)

        # Multiply the 2D matrices using the C++ function
        result_list = [matmul_cpp(np.matrix(mat1), np.matrix(mat2)) for mat1, mat2 in zip(list1, list2)]

        # Convert the result list back to a numpy array and reconstruct the n-dimensional array
        result_array = reconstruct_nd_array_from_2d_list(result_list, array1.shape)
        return heatflow.Tensor(result_array)
    if array1.ndim == 1:
        return heatflow.Tensor(sum(a * b for a, b in zip(array1, array2)))
    else:
        return heatflow.Tensor(matmul_cpp(array1, array2))
    
def add(array1, array2):
    """
    Adds two n-dimensional numpy arrays using Eigen's matrix multiplication.
    """

    if isinstance(array1, heatflow.Tensor):
        array1 = array1.data
    
    if isinstance(array2, heatflow.Tensor):
        array2 = array2.data
        
    if array1.shape != array2.shape:
        raise ValueError("The two arrays must have the same shape.")
    
    # Flatten the n-dimensional arrays to lists of 2D arrays
    if array1.ndim > 2:
        list1 = flatten_nd_array_to_2d_list(array1)
        list2 = flatten_nd_array_to_2d_list(array2)

        # Multiply the 2D matrices using the C++ function
        result_list = [add_cpp(np.matrix(mat1), np.matrix(mat2)) for mat1, mat2 in zip(list1, list2)]

        # Convert the result list back to a numpy array and reconstruct the n-dimensional array
        result_array = reconstruct_nd_array_from_2d_list(result_list, array1.shape)
        return heatflow.Tensor(result_array)
    elif array1.ndim == 1:
        return heatflow.Tensor(array1 + array2)
    else:
        return heatflow.Tensor(add_cpp(array1, array2))

def subtract(array1, array2):
    """
    Subtracts two n-dimensional numpy arrays using Eigen's matrix multiplication.
    """

    if isinstance(array1, heatflow.Tensor):
        array1 = array1.data
    
    if isinstance(array2, heatflow.Tensor):
        array2 = array2.data
        
    if array1.shape != array2.shape:
        raise ValueError("The two arrays must have the same shape.")
    
    # Flatten the n-dimensional arrays to lists of 2D arrays
    if array1.ndim > 2:
        list1 = flatten_nd_array_to_2d_list(array1)
        list2 = flatten_nd_array_to_2d_list(array2)

        # Multiply the 2D matrices using the C++ function
        result_list = [subtract_cpp(np.matrix(mat1), np.matrix(mat2)) for mat1, mat2 in zip(list1, list2)]

        # Convert the result list back to a numpy array and reconstruct the n-dimensional array
        result_array = reconstruct_nd_array_from_2d_list(result_list, array1.shape)
        return heatflow.Tensor(result_array)
    elif array1.ndim == 1:
        return heatflow.Tensor(array1 - array2)
    else:
        return heatflow.Tensor(subtract_cpp(array1, array2))
    
def divide(array1, array2):
    """
    Divides two n-dimensional numpy arrays using Eigen's matrix multiplication.
    """

    if isinstance(array1, heatflow.Tensor):
        array1 = array1.data
    
    if isinstance(array2, heatflow.Tensor):
        array2 = array2.data
        
    if array1.shape != array2.shape:
        raise ValueError("The two arrays must have the same shape.")
    
    # Flatten the n-dimensional arrays to lists of 2D arrays
    if array1.ndim > 2:
        list1 = flatten_nd_array_to_2d_list(array1)
        list2 = flatten_nd_array_to_2d_list(array2)

        # Multiply the 2D matrices using the C++ function
        result_list = [divide_cpp(np.matrix(mat1), np.matrix(mat2)) for mat1, mat2 in zip(list1, list2)]

        # Convert the result list back to a numpy array and reconstruct the n-dimensional array
        result_array = reconstruct_nd_array_from_2d_list(result_list, array1.shape)
        return heatflow.Tensor(result_array)
    elif array1.ndim == 1:
        return heatflow.Tensor(array1 / array2)
    else:
        return heatflow.Tensor(divide_cpp(array1, array2))