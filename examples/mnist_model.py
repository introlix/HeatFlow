import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import heatflow
import heatflow.nn as nn
import heatflow.nn.functional as F
from heatflow.optim.sgd import SGD
from heatflow.tools.datasets import mnist

from tqdm import trange

x_train, y_train, x_test, y_test = mnist.getMNIST()

class MnistModel(nn.Module):
    def __init__(self):
        self.l1 = nn.Linear(28 * 28, 128, bias=False)
        self.l2 = nn.Linear(128, 10, bias=False)

    def forward(self, x) -> heatflow.Tensor:
        linear1 = self.l1(x)
        linear2 = self.l2(linear1)
        out = F.softmax(linear2)

        return out

model = MnistModel()
EPOCHS = 1000
batch_size = 1000
lr = 0.01

optimizer = SGD(parameters=model.parameters(), lr=lr)
t_bar = trange(EPOCHS)

losses = []

for epoch in t_bar:
    optimizer.zero_grad()

    ids = np.random.choice(60000, batch_size)
    x = heatflow.Tensor(x_train[ids])
    y = heatflow.Tensor(y_train[ids])

    y_pred = model(x)
    diff = y - y_pred
    loss = (diff ** 2).sum() * (1.0 / diff.shape[0])

    loss.backward()

    optimizer.step()
    losses.append(loss.data)

    t_bar.set_description("Epoch: %.0f Loss: %.8f" % (epoch, loss.data))

y_pred = model(heatflow.Tensor(x_test))

acc = np.array(np.argmax(y_pred.data, axis=1) == np.argmax(y_test, axis=1)).sum()
print("Accuracy: ", acc / len(x_test))