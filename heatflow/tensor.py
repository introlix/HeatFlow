from typing import List, Tuple, Union
import numpy as np

from .utils import InputError, RequiresGradError

ArrayableType = Union[float, list, np.ndarray]
TensorableType = Union[float, np.ndarray, "Tensor"]

# TODO: argmax
# TODO: Slice, transpose, resize


class Tensor:
    """
    Stores data for training

    Parameters:
        data (int, float, list, np.ndarray): The data that will be stored in Tensor or manipulated
        requires_grad (bool): Whether the Tensor requires gradient to be calculated. By default it is False.
        grad (np.array): The gradient value of the Tensor. By default it is zero when requires_grad is True else None
        ctx (List[Tensor]): list of all the operand tensors which resulted to this tensor
        backward_fn (Callable[[], None]): reference to the function to calculate the gradient for the operand tensors
    """

    def __init__(
        self, data: ArrayableType = None, requires_grad: bool = False, dtype=np.float64
    ) -> None:
        
        """
        Args:
            data (int, float, list, np.ndarray): The data that will be stored in Tensor or manipulated
            requires_grad (bool): Whether the Tensor requires gradient to be calculated. By default it is False.
            grad (np.array): The gradient value of the Tensor. By default it is zero when requires_grad is True else None
        """

        self._data = enforceNumpy(data, dtype=dtype)
        self.ctx: List["Tensor"] = []
        self.grad = Tensor(np.zeros_like(self.data)) if requires_grad == True else None
        self.backward_fn = lambda: None
        self.requires_grad = requires_grad

    def save_for_backward(self, inputs: List["Tensor"]) -> None:
        """Stores the tensors used to compute `self`"""
        self.ctx += inputs

    def init_gradient(self, gradient: Union[float, "Tensor"]) -> None:
        """Init the gradient of this tensor"""

        if self.data.size != 1 and gradient == 1.0:
            if gradient is None:
                raise ValueError(
                    "Default backward function can only be computed for scalar values. Pass `gradient` for vector outputs"
                )

        self.grad = enforceTensor(gradient)

    def generate_computational_graph(self) -> List["Tensor"]:
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

    def backward(self, gradient: Union[float, "Tensor"] = 1.0) -> None:
        """Traverses through the computational graph to compute gradients

        Parameters
        ----------

        Arg: gradient (Tensor)
            Gradient of the output tensor w.r.t to itself

        """

        if self.requires_grad is False:
            raise RequiresGradError(
                self,
                "Tensors does not require grad. Enable requires_grad to compute gradients",
            )

        self.init_gradient(gradient)
        gradient_tape = self.generate_computational_graph()

        for v in reversed(gradient_tape):
            v.backward_fn()

    def zero_grad(self) -> None:
        if self.requires_grad is False:
            raise RequiresGradError(
                self,
                "Tensors does not require grad. Enable requires_grad to compute gradients",
            )
        self.grad = Tensor(np.zeros_like(self.data))

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, data) -> None:
        self._data = data
        self.requires_grad = False

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

    # Internal operators

    def dot(self, x: TensorableType) -> "Tensor":
        return self.matmul(x)

    # Generate functions

    @staticmethod
    def uniform(
        low: int = 0.0,
        high: int = 1.0,
        size: Tuple[int] = None,
        requires_grad: bool = False,
    ) -> "Tensor":
        return Tensor(
            np.random.uniform(low, high, size=size), requires_grad=requires_grad
        )
    
    @staticmethod
    def zeros_like(x: ArrayableType) -> "Tensor":
        if isinstance(x, Tensor):
            return Tensor(np.zeros_like(x.data))

        x = enforceNumpy(x)
        return Tensor(np.zeros_like(x))

    @staticmethod
    def ones_like(x: ArrayableType) -> "Tensor":
        if isinstance(x, Tensor):
            return Tensor(np.ones_like(x.data))

        x = enforceNumpy(x)
        return Tensor(np.ones_like(x))

    @staticmethod
    def randn(*dim, requires_grad: bool = False) -> "Tensor":
        """Returns random floating-point tensor

        Example
        -------
        >> a = matterix.randn(2,2)
        Tensor([[ 0.36658869 -1.2298281 ]
        [ 1.63282169 -0.669039  ]], shape=(2, 2))

        """
        return Tensor(np.random.randn(*dim), requires_grad=requires_grad)

    @staticmethod
    def randint(
        size: Tuple[int], low: int, high: int = None, requires_grad: bool = False
    ) -> "Tensor":
        """Returns random floating-point tensor

        Example
        -------
        >> a = matterix.randint(0,2 (3,3))
        Tensor([[ 1 0 1],
        [ 2 2]], shape=(3, 3))

        """
        return Tensor(
            np.random.randint(low, high, size=size), requires_grad=requires_grad
        )

    # @staticmethod
    # def eye(rows: int, columns: int) -> "Tensor":
    #     """Returns identity tensor

    #     Parameters
    #     ----------

    #     Arg: rows (int)
    #         Number of rows in the tensor

    #     Arg: columns (int)
    #         Number of columns in the tensor

    #     """
    #     return Tensor(np.eye(int(rows), int(columns)))

    # Conversion and meta functions

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

    # Todo: Some error need to handle
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

    def tolist(self) -> List[float]:
        """Returns tensor as a list"""
        return self.data.tolist()

    def numel(self) -> int:
        """Returns the number of elements in a tensor

        Example
        -------

        >> a = Tensor([[1,2,3], [4,5,6], [7,8,9]])
        >> a.shape
        (3,3)
        >> a.numel() # 9, as there are 9 elements in the tensor
        9
        """
        return np.prod(self.shape)

    def __repr__(self) -> str:
        return f"Tensor({self.data}, req_grad={self.requires_grad}, shape={self.shape})"


def enforceTensor(_input: TensorableType) -> "Tensor":
    """Converts input to tensor. This is called whenever an operation is performed"""
    if isinstance(_input, Tensor) is True:
        return _input
    else:
        return Tensor(_input)


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