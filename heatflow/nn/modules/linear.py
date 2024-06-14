from .module import Module
from heatflow import Tensor
from heatflow.nn import Parameter
import numpy as np

class Linear(Module):
    def __init__(self, input_dim, output_dim, bias: bool = False) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dims = (input_dim, output_dim)

        if output_dim is None:
            self.dims = (input_dim, )
        
        self.w = Parameter(
            np.random.uniform(-1, 1, size=self.dims) / np.sqrt(np.prod(self.dims)),
            requires_grad=True,
        )

        self.init_bias(set_bias = bias)

    def init_bias(self, set_bias: bool):
        if set_bias:
            if self.output_dim is None:
                self.b = Parameter.zeros_like(1, requires_grad=True)
            else:
                self.b = Parameter(np.zeros((1, self.output_dim)), requires_grad=True)
        else:
            self.b = Parameter(0.0)
        
    def forward(self, x) -> Tensor:

        output = x @ self.w + self.b

        return output