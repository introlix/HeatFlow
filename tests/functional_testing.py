import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import heatflow
import heatflow.nn as nn
import heatflow.nn.functional as F

a = heatflow.Tensors([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
sigmoid_value = F.sigmoid(a)
relu_value = F.relu(a)
softmax_value = F.softmax(a)
tanh_value = F.softmax(a)