class SGD:
    """
    Stochastic gradient descent for optimization

    Args:
        model (Module): model which need to be optimize
        parameters (dict): dict of all the parameter that need to get optimized, i.e. weight and bias.
        lr (float): the size of the gradient steps
        momentum (float): the value of the momentum
    """
    def __init__(self, parameters: dict, lr: float = 0.001, momentum: float = 0.9) -> None:
        self.params = parameters
        self.lr = lr
        self.momentum = momentum

        self.momentum_dict = {param: 0.0 for param, _ in self.params.items()}

    def step(self) -> None:
        """Updates the parameters of the model"""
        for param, weight in self.params.items():
            v = (self.momentum * self.momentum_dict[param]) + (self.lr * weight.grad.data)
            weight -= v

            self.params[param] = weight
            self.momentum_dict[param] = v

    def zero_grad(self) -> None:
        """Sets the gradients of all the parameters to zero"""

        for _, param in self.params.items():
            param.zero_grad()