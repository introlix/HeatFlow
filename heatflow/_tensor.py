import numpy as np
from typing import List, Tuple, Union

from .utils import InputError, RequiresGradError

ArrayableType = Union[float, list, np.ndarray]
TensorableType = Union[float, np.ndarray, "Tensors"]

class Tensors:
    """
    Stores data for training

    Parameters:
        data (int, float, list, np.ndarray): The data that will be stored in Tensor or manipulated
        requires_grad (bool): Whether the Tensor requires gradient to be calculated. By default it is False.
        grad (np.array): The gradient value of the Tensor. By default it is zero when requires_grad is True else None
    """
    def __init__(self, data: ArrayableType = None, requires_grad: bool = False, dtype=np.float64) -> None:
        """
        Args:
            data (int, float, list, np.ndarray): The data that will be stored in Tensor or manipulated
            requires_grad (bool): Whether the Tensor requires gradient to be calculated. By default it is False.
            grad (np.array): The gradient value of the Tensor. By default it is zero when requires_grad is True else None
        """

        self._data = enforceNumpy(data, dtype=dtype)
        self.ctx: List["Tensors"] = []
        self.grad = Tensors(np.zeros_like(self.data)) if requires_grad == True else None
        self.grad_fn = lambda: None
        self.requires_grad = requires_grad

    def save_for_backward(self, inputs: List["Tensors"]) -> None:
        """Stores the tensors used to compute `self`"""
        self.ctx += inputs

    def init_gradient(self, gradient: Union[float, "Tensors"]) -> None:
        """Init the gradient of this tensor"""

        if self.data.size != 1 and gradient == 1.0:
            if gradient is None:
                raise ValueError(
                    "Default backward function can only be computed for scalar values. Pass `gradient` for vector outputs"
                )

        self.grad = enforceTensor(gradient)

    def generate_computational_graph(self) -> List["Tensors"]:
        """Performs topological sorting on the operations"""

        gradient_tape = list()
        visited = set()

        # This part of the topological sort is from https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.ctx:
                    build_topo(child)
                gradient_tape.append(v)

        build_topo(self)

        return gradient_tape
    
    def backward(self, gradient: Union[float, "Tensors"] = 1.0) -> None:
        """Traverses through the computational graph to compute gradients

        Arg:
            gradient (Tensor): Gradient of the output tensor w.r.t to itself

        """

        if self.requires_grad is False:
            raise RequiresGradError(
                self,
                "Tensors does not require grad. Enable requires_grad to compute gradients",
            )
        
        self.init_gradient(gradient)
        gradient_tape = self.generate_computational_graph()

        for v in reversed(gradient_tape):
            v.grad_fn()
    
    def zero_grad(self) -> None:
        if self.requires_grad is False:
            raise RequiresGradError(
                self,
                "Tensors does not require grad. Enable requires_grad to compute gradients",
            )
        self.grad = Tensors(np.zeros_like(self.data))


    @property
    def data(self) -> np.ndarray:
        return self._data
    
    @data.setter
    def data(self, data) -> None:
        self._data = data
        self.requires_grad = False

    # def data(self, data):
    #     '''Sets the data to the Tensor

    #     Args:
    #         data (int or float or list or np.ndarray): Data to be set
    
    #     Raises:
    #         TypeError: If data is not instance of (int or float or list or np.ndarray)
    #     '''
    #     self._data = process_data(data)

    def dot(self, x: TensorableType) -> "Tensors":
        return self.matmul(x)
    
    @staticmethod
    def uniform(
        low: int = 0.0,
        high: int = 1.0,
        size: Tuple[int] = None,
        requires_grad: bool = False,
    ) -> "Tensors":
        return Tensors(
            np.random.uniform(low, high, size=size), requires_grad=requires_grad
        )
    
    @staticmethod
    def zeros_like(x: ArrayableType) -> "Tensors":
        if isinstance(x, Tensors):
            return Tensors(np.zeros_like(x.data))

        x = enforceNumpy(x)
        return Tensors(np.zeros_like(x))
    
    
    @staticmethod
    def ones_like(x: ArrayableType) -> "Tensors":
        if isinstance(x, Tensors):
            return Tensors(np.ones_like(x.data))

        x = enforceNumpy(x)
        return Tensors(np.ones_like(x))
    
    @staticmethod
    def randn(*dim, requires_grad: bool = False) -> "Tensors":
        """Returns random floating-point tensor
        """
        return Tensors(np.random.randn(*dim), requires_grad=requires_grad)
    
    @staticmethod
    def randint(
        size: Tuple[int], low: int, high: int = None, requires_grad: bool = False
    ) -> "Tensors":
        """Returns random floating-point tensor
        """
        return Tensors(
            np.random.randint(low, high, size=size), requires_grad=requires_grad
        )
        
    def reshape(self, *newshape):
        """
        Reshape tensor to a new shape.

        Returns:
            Tensor: A new tensor with new shape
        """
        return Tensors(np.reshape(self.data, newshape))
    
    def flatten(self):
        """
        Flatten tensor to 1D

        Returns:
            Tensor: A new 1D Tensor
        """
        return Tensors(self.data.flatten())
    
    def expand_dims(self, axis):
        """
        Expand the dimensions of a given tensor by adding a new axis at the specified position.

        Args:
            axis (int): The position at which to insert the new axis.

        Returns:
            Tensor: The matrix with the expanded dimensions.
        """
        return Tensors(np.expand_dims(self.data, axis))

    def squeeze(self, axis=None):
        """
        Remove single-dimensional entries from the shape of a given tensor.

        Args:
            axis (int or tuple of int, optional): Selects a subset of the single-dimensional entries in the shape. 
                                            If an axis is selected with shape entry greater than one, an error is raised.

        Returns:
            Tensor: The tensor with single-dimensional entries removed.
        """
        return Tensors(np.squeeze(self.data, axis=axis))
    

    @property
    def shape(self) -> Tuple:
        """Returns the shape of the tensor"""

        if self.data.shape == ():
            return (1,)
        return self.data.shape
    
    @property
    def ndim(self) -> int:
        """Returns the rank of the tensor"""
        return self.data.ndim
    

    def tolist(self):
        """Returns tensor as a list"""
        return self.data.tolist()
    
        
    def __repr__(self):
        return f'Tensor({self.data}, requires_grad={self.requires_grad})'
    
    def __str__(self):
        return f'Tensor({self.data}, requires_grad={self.requires_grad}, shape={self.shape})\n'
    

def enforceTensor(_input):
        """
        Converts teh input into tensor.
        """
        if isinstance(_input, Tensors) is True:
            return _input
        else:
            return Tensors(_input)
        
def enforceNumpy(_input: ArrayableType, dtype=np.float64) -> np.ndarray:
    """Converts the input to numpy array. This is called only during input validation"""

    if _input is None:
        raise InputError(_input, "No input data provided. Tensor cannot be empty.")

    if not isinstance(_input, np.ndarray):
        if type(_input) in [
            list,
            float,
            np.float32,
            np.float64,
            np.float16,
            np.float128,
        ]:
            return np.array(_input, dtype=dtype)
        raise InputError(
            _input, "Tensor only accepts float, list and numpy array as data."
        )

    _input = _input.astype(dtype)

    return _input