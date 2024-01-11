from doctest import OutputChecker
from utils import *
import torch.nn.functional as F

class Graph_Linear(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(Graph_Linear, self).__init__()
        self.bias = bias
        self.W = nn.Parameter(torch.zeros(input_size,hidden_size))
        self.b = nn.Parameter(torch.zeros(hidden_size))
        self.reset_parameters()
    def reset_parameters(self):
        reset_parameters(self.named_parameters)
    def forward(self, x):
        output = torch.mm(x, self.W)
        if self.bias:
            output = output + self.b
        return output

class Graph_GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(Graph_GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = Graph_Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = Graph_Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.reset_parameters()
    def reset_parameters(self):
        reset_parameters(self.named_parameters)
    def forward(self, x, hidden):
        gate_x = self.x2h(x)
        gate_h = self.h2h(hidden)
        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()
        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + (resetgate * h_n))
        hy = newgate + inputgate * (hidden - newgate)
        return hy

class Graph_GRUModel(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim, bias=True):
        super(Graph_GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.gru_cell = Graph_GRUCell(input_dim, hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        reset_parameters(self.named_parameters)

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = torch.zeros(self.num_nodes, self.hidden_dim, device=x.device,dtype = x.dtype)
        for seq in range(x.size(0)):
            hidden = self.gru_cell(x[seq], hidden)
        return hidden