import numpy as np
# from ops.basics import add, subtract, divide
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
        self.grad = Tensor(np.zeros_like(self.data)) if requires_grad == True else None
        self.grad_fn = lambda: None
        self.ctx = []

    def save_for_backward(self, inputs) -> None:
        """Stores the tensors used to compute `self`"""
        self.ctx += inputs

    def init_gradient(self, gradient):
        """Init the gradient of this tensor"""
        
        if self.data.size != 1 and gradient == 1.0:
            if gradient is None:
                raise ValueError(
                    "Default backward function can only be computed for scalar values. Pass `gradient` for vector outputs"
                )
            
        self.grad = toTensor(gradient)

    def generate_computational_graph(self):
        """Performs topological sorting on the operations"""

        gradient_tape = list()
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.ctx:
                    build_topo(child)
                gradient_tape.append(v)

        build_topo(self)
        return gradient_tape
    
    def backward(self, gradient = 1.0):
        """Traverses through the computational graph to compute gradients

        Arg:
            gradient (Tensor): Gradient of the output tensor w.r.t to itself

        """

        if self.requires_grad is False:
            raise ValueError("Tensor does not require grad. To compute gradients enable requires_grad")
        
        self.init_gradient(gradient)
        gradient_tape = self.generate_computational_graph()

        for v in reversed(gradient_tape):
            v.grad_fn()
    
    def zero_grad(self):
        if self.requires_grad is False:
            raise ValueError("Tensor does not require grad. To compute gradients enable requires_grad")
        
        self.grad = Tensor(np.zeros_like(self.data))
    
    def set_grad_fn(self, grad_fn):
        self.grad_fn = grad_fn if self.requires_grad else None
    
    def ones_like(x):
        if isinstance(x, Tensor):
            return Tensor(np.ones_like(x.data))

        x = toTensor(x)
        return Tensor(np.ones_like(x))
        
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

    @property    
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):
        return self.data.ndim
    

    def tolist(self):
        """Returns tensor as a list"""
        return self.data.tolist()
    
        
    def __repr__(self):
        return f'Tensor({self.grad}, requires_grad={self.requires_grad})'
    
    def __str__(self):
        return f'Tensor({self.data}, requires_grad={self.requires_grad}, shape={self.shape})\n'
    

def toTensor(_input):
        """
        Converts teh input into tensor.
        """
        if isinstance(_input, Tensor) is True:
            return _input
        else:
            return Tensor(_input)