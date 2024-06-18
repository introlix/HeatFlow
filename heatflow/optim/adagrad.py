import numpy as np
from heatflow.tensor import Tensor

class AdaGrad:
    def __init__(self, parameters: dict, lr: float = 0.001, epsilon: float = 1e-8) -> None:
        self.params = parameters
        self.epsilon = epsilon
        self.lr = lr

        # Initialize squared gradient sum dictionary
        self.squared_gradients = {param: Tensor.zeros_like(weight) for param, weight in self.params.items()}

    def step(self) -> None:
        """Updates the parameters of the model"""

        for param, weight in self.params.items():
            grad = weight.grad

            # Update the sum of squared gradients
            self.squared_gradients[param] += grad ** 2

            # Compute the adjusted learning rate
            adjusted_lr = self.lr / ((self.squared_gradients[param] ** 0.5) + self.epsilon)

            # Update the weights
            weight -= adjusted_lr * grad

            # Update the parameter in the params dictionary
            self.params[param] = weight

    def zero_grad(self) -> None:
        """Sets the gradients of all the parameters to zero"""

        for _, param in self.params.items():
            param.zero_grad()
