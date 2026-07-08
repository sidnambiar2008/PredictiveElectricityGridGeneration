import torch
from torch import nn
import numpy as np




class EnergyLSTM(nn.Module):
    def __init__(self, num_inputs, num_hidden):
        super().__init__()
