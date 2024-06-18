import inspect
from abc import ABC, abstractmethod
from typing import Dict
from heatflow.tensor import Tensor

class Module(ABC):
    """Abstract base class to define neural networks"""

    @abstractmethod
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Implement forward pass"""
        raise NotImplementedError("Forward function is not implemented")

    def parameters(self) -> Dict[str, Tensor]:
        """Returns all optimizable parameters"""
        params = dict()

        def filter_condition(x) -> bool:
            """Checks if object is a parameter"""

            if (isinstance(x, Tensor) and x.requires_grad == True) or isinstance(
                x, Module
            ):
                return True

            return False

        def add_prefix_to_keys(prefix: str, children_params: dict) -> dict:
            return {".".join([prefix, k]): v for k, v in children_params.items()}

        # only if predicate is true, it is included in the list
        for obj_name, obj in inspect.getmembers(self, predicate=filter_condition):
            if isinstance(obj, Module):
                # get all the parameters from that module and add prefix
                children_parameters = add_prefix_to_keys(obj_name, obj.parameters())
                params.update(children_parameters)
                continue

            params[obj_name] = obj

        return params

    def zero_grad(self) -> None:
        """Zeros out the gradient buffers of all optimizable parameters"""
        params = self.parameters()
        for k, v in params.items():
            v.zero_grad()
            params[k] = v

        self.__dict__.update(params)

    def __call__(self, x) -> Tensor:
        return self.forward(x)