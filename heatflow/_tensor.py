import numpy as np
from ops.basics import matmul, add, subtract, divide
from utils import process_data

class Tensor:
    """
    Stores data for training

    Parameters:
        data (int, float, list, np.ndarray): The data that will be stored in Tensor or manipulated
        requires_grad (bool): Whether the Tensor requires gradient to be calculated. By default it is False.
        grad (np.array): The gradient value of the Tensor. By default it is zero when requires_grad is True else None
    """
    def __init__(self, data, requires_grad=False) -> None:
        """
        Args:
            data (int, float, list, np.ndarray): The data that will be stored in Tensor or manipulated
            requires_grad (bool): Whether the Tensor requires gradient to be calculated. By default it is False.
            grad (np.array): The gradient value of the Tensor. By default it is zero when requires_grad is True else None
        """

        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = None

        if requires_grad:
            self.zero_grad()
    
    def zero_grad(self):
        self.grad = np.zeros_like(self.data, dtype=np.float32)
    
    def set_grad_fn(self, grad_fn):
        self.grad_fn = grad_fn if self.requires_grad else None

    def __add__(self, other):
        if isinstance(other, Tensor):
            return add(self.data, other.data)
        else:
            return add(self.data, other)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return subtract(self.data, other.data)
        else:
            return subtract(self.data, other)

    def __matmul__(self, other):
        if isinstance(other, Tensor):
            return matmul(self.data, other.data)
        else:
            return matmul(self.data, other)
        
    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return divide(self.data, other.data)
        else:
            return divide(self.data, other)
        
    def reshape(self, *newshape):
        """
        Reshape tensor to a new shape.

        Returns:
            Tensor: A new tensor with new shape
        """
        return Tensor(np.reshape(self.data, newshape))
    
    def flatten(self):
        """
        Flatten tensor to 1D

        Returns:
            Tensor: A new 1D Tensor
        """
        return Tensor(self.data.flatten())
    
    def expand_dims(self, axis):
        """
        Expand the dimensions of a given tensor by adding a new axis at the specified position.

        Args:
            axis (int): The position at which to insert the new axis.

        Returns:
            Tensor: The matrix with the expanded dimensions.
        """
        return Tensor(np.expand_dims(self.data, axis))

    def squeeze(self, axis=None):
        """
        Remove single-dimensional entries from the shape of a given tensor.

        Args:
            axis (int or tuple of int, optional): Selects a subset of the single-dimensional entries in the shape. 
                                            If an axis is selected with shape entry greater than one, an error is raised.

        Returns:
            Tensor: The tensor with single-dimensional entries removed.
        """
        return Tensor(np.squeeze(self.data, axis=axis))
        
    def data(self, data):
        '''Sets the data to the Tensor

        Args:
            data (int or float or list or np.ndarray): Data to be set
    
        Raises:
            TypeError: If data is not instance of (int or float or list or np.ndarray)
        '''
        self._data = process_data(data)
        
    def shape(self):
        return self.data.shape
        
    def __repr__(self):
        return f'Tensor({self.grad}, requires_grad={self.requires_grad})'
    
    def __str__(self):
        return f'Tensor({self.data}, requires_grad={self.requires_grad}, shape={self.shape()})\n'