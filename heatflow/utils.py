import numpy as np
from numba import jit
from functools import wraps

class InputError(ValueError):
    """Custom error that is raised for invalid input to Tensor"""

    def __init__(self, _object: any, message: str) -> None:
        self.object = _object
        self.message = message
        super().__init__(message)

class RequiresGradError(RuntimeError):
    def __init__(self, _object: any, message: str) -> None:
        self.object = _object
        self.message = message
        super().__init__(message)

@jit(nopython=True)
def process_data(data):
    """
    Checks the type of data and convert it in supported data types

    supported types: [int, float, list, np.ndarray]

    Args:
        data (int, float, list, np.ndarray): data to be preprocess

    Returns:
        Processed data

    Raises:
        TypeError: If data is not float or typecastable to float
        TypeError: If data type in not supported
    """
    supported_types = (int, float, list, np.ndarray)

    if type(data) in supported_types:
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        try:
            data = data.astype(float)
        except ValueError:
            raise TypeError("Elements of data should be of type float or be typecastable to float")
    else:
        raise TypeError(f"Expected data of types {supported_types} instead got {type(data)}")
    return data

def registerFn(cls, fn_name):
    """Decorator to add function dynamically to a class"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        setattr(cls, fn_name, wrapper)
        return func

    return decorator

def flatten_nd_array_to_2d_list(array):
    """
    Flattens an n-dimensional numpy array to a list of 2D arrays.

    Args:
        array (np.ndarray): nd array to be flattend into 2d list
    
    Returns:
        List of 2d arrays
    """
    shape = array.shape
    num_matrices = np.prod(shape[:-2])
    reshaped_array = array.reshape(num_matrices, shape[-2], shape[-1])
    return [reshaped_array[i] for i in range(num_matrices)]

def reconstruct_nd_array_from_2d_list(flat_list, original_shape):
    """
    Reconstructs an n-dimensional numpy array from a list of 2D arrays.

    Args:
        flat_list (list of np.array): A list of 2D numpy arrays to be reconstructed.
        original_shape (tuple): The shape of the original n-dimensional array.

    Returns:
        ND_Array: The reconstructed n-dimensional array.
    """
    new_shape = original_shape[:-2] + (flat_list[0].shape[0], flat_list[0].shape[1])
    flat_array = np.array(flat_list)
    return flat_array.reshape(new_shape)

def to_categorical(x):

    a = x.flatten()

    one_hot = np.zeros((a.size, a.max() + 1))
    rows = np.arange(a.size)

    one_hot[rows, a] = 1

    return one_hot

def underDevelopment(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        raise NotImplementedError("Function still under development")

    return wrapper