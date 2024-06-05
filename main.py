from build.heatflow import add, matmul
import numpy as np
import time

def flatten_nd_array_to_2d_list(array):
    """
    Flattens an n-dimensional numpy array to a list of 2D arrays.
    """
    shape = array.shape
    num_matrices = np.prod(shape[:-2])
    reshaped_array = array.reshape(num_matrices, shape[-2], shape[-1])
    return [reshaped_array[i] for i in range(num_matrices)]

def reconstruct_nd_array_from_2d_list(flat_list, original_shape):
    """
    Reconstructs an n-dimensional numpy array from a list of 2D arrays.
    """
    new_shape = original_shape[:-2] + (flat_list[0].shape[0], flat_list[0].shape[1])
    flat_array = np.array(flat_list)
    return flat_array.reshape(new_shape)

def multiply_nd_arrays(array1, array2):
    """
    Multiplies two n-dimensional numpy arrays using Eigen's matrix multiplication.
    """
    if array1.shape != array2.shape:
        raise ValueError("The two arrays must have the same shape.")

    # Flatten the n-dimensional arrays to lists of 2D arrays
    list1 = flatten_nd_array_to_2d_list(array1)
    list2 = flatten_nd_array_to_2d_list(array2)

    # Convert lists of numpy arrays to lists of Eigen matrices
    eigen_list1 = [np.matrix(mat) for mat in list1]
    eigen_list2 = [np.matrix(mat) for mat in list2]

    # Multiply the 2D matrices using the C++ function
    result_list = matmul(eigen_list1, eigen_list2)

    # Convert the result list back to a numpy array and reconstruct the n-dimensional array
    result_array = np.array(result_list)
    return reconstruct_nd_array_from_2d_list(result_list, array1.shape)

a = np.random.rand(10, 10, 10, 10)
b = np.random.rand(10, 10, 10, 10)

start_time = time.time()
result = multiply_nd_arrays(a, b)
end_time = time.time()
