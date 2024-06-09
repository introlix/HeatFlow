from heatflow_cpp import add_cpp, matmul_cpp
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

def matmul(array1, array2):
    """
    Multiplies two n-dimensional numpy arrays using Eigen's matrix multiplication.
    """
    if array1.shape != array2.shape:
        raise ValueError("The two arrays must have the same shape.")

    # Flatten the n-dimensional arrays to lists of 2D arrays
    if array1.ndim > 2:
        list1 = flatten_nd_array_to_2d_list(array1)
        list2 = flatten_nd_array_to_2d_list(array2)

        # Multiply the 2D matrices using the C++ function
        result_list = [matmul_cpp(np.matrix(mat1), np.matrix(mat2)) for mat1, mat2 in zip(list1, list2)]

        # Convert the result list back to a numpy array and reconstruct the n-dimensional array
        result_array = reconstruct_nd_array_from_2d_list(result_list, array1.shape)
        return result_array
    if array1.ndim == 1:
        return sum(a * b for a, b in zip(array1, array2))
    else:
        return matmul_cpp(array1, array2)

a = np.array([[[1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3]]])
b = np.array([[[1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3]]])


# start_time = time.time()
result = matmul(a, b)
# end_time = time.time()
# print(np.matmul(a, b))
# print(result)

print(result)

# print(f"Time taken: {end_time - start_time} seconds")