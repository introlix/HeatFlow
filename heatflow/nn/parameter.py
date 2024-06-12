import heatflow

class Parameter(heatflow.Tensor):
    """
    A kind of Tensor that is to be considered a module parameter.

    Parameters are Tensor subclasses, that have a special property when used
    with modules - when they're assigned as Module attributes they are
    automatically added to the list of its parameters, and will appear e.g.
    in parameters() iterator. This is typically used to represent the learnable
    weights of a neural network.

    Attributes:
        data (tensor, np.ndarray or list or float): The initial value of the parameter.
        requires_grad (bool): If True, the parameter will be involved in the gradient computation.
    """
    def __init__(self, data, requires_grad=True):
        if isinstance(data, heatflow.Tensor):
            data = data.data
        super().__init__(data, requires_grad)

    def __repr__(self):
        return f"Parameter: tensor({self.data}, requires_grad={self.requires_grad})"
    
    def __str__(self):
        return f"Parameter:\ntensor({self.data}, requires_grad={self.requires_grad})"