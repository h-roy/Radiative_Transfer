import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader
from torchvision import transforms


class PINN(nn.Module):

    def __init__(self, network_depth, network_width):
        super(PINN, self).__init__()
        self.input_layer = nn.Linear(3, network_width)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(network_width, network_width) for _ in range(network_depth - 1)])
        self.output_layer = nn.Linear(network_width, 2)

    def forward(self, x):
        x = nn.Tanh(self.input_layer(x))
        for l in self.hidden_layers:
            x = nn.Tanh(l(x))
        x = self.output_layer(x)
        return x

def init_xavier(model):
    def init_weights(m):
        if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
            gain = nn.init.calculate_gain('tanh')
            # gain = 1
            torch.nn.init.xavier_uniform_(m.weight, gain=gain)
            m.bias.data.fill_(0.0)

    model.apply(init_weights)
