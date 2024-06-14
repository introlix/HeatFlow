from heatflow import Tensor, toTensor

# Loss functions
def MSE(y_train: Tensor, y_pred: Tensor, norm: bool = True) -> Tensor:
    """
    Function to find mean squared error for the model predictions

    Args:
        y_train (Tensor): the actual value
        y_pred (Tensor): the predicted value
    """

    diff = y_train - y_pred
    loss = (diff * diff).sum() * (1.0 / diff.shape[0]) if norm else (diff * diff).sum()

    return loss

# Activation functions
def softmax(x) -> Tensor:
    x = toTensor(x)
    return x.softmax()

def relu(x) -> Tensor:
    x = toTensor(x)
    return x.relu()

def sigmoid(x) -> Tensor:
    x = toTensor(x)
    return x.sigmoid()

def tanh(x) -> Tensor:
    x = toTensor(x)
    return x.tanh()