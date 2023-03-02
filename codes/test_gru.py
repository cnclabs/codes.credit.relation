import torch
import torch.nn as nn
import torch.nn.functional as F
from doctest import OutputChecker
import math
from torch import optim
from torchinfo import summary

from Layers import *
from model import *
from utils import *
from doctest import OutputChecker
from utils import *
import torch.nn.functional as F

json_path = '/tmp2/cwlin/explainable_credit/explainable_credit/experiments/gru01_index/index_fold_01/params.json'
params = Params(json_path)

NUM_STOCK=15786
DEVICE = torch.device(f"cuda:1")

tensor_input = torch.load('/tmp2/cwlin/explainable_credit/graph_gru_sentence.pt', map_location=lambda storage, loc: storage.cuda(1))
print(tensor_input.shape) # at 1 time t, 15786 companies, 14 features

class Graph_Linear(nn.Module):
    def __init__(self,num_nodes, input_size, hidden_size, bias=True):
        super(Graph_Linear, self).__init__()
        self.bias = bias
        self.W = nn.Parameter(torch.zeros(num_nodes,input_size,hidden_size))
        self.b = nn.Parameter(torch.zeros(num_nodes,hidden_size))
        self.reset_parameters()
    def reset_parameters(self):
        reset_parameters(self.named_parameters)
    def forward(self, x):
        # print("x: {}".format(x.shape))
        output = torch.bmm(x.unsqueeze(1), self.W) # (b, n, m) x (b, m, p) -> (b, n, p)
        # print("output: {}".format(output.shape))
        output = output.squeeze(1)
        # print("output: {}".format(output.shape))
        if self.bias:
            output = output + self.b
        # print("output: {}".format(output.shape))
        return output

class Graph_GRUCell(nn.Module):
    def __init__(self, num_nodes, input_size, hidden_size, bias=True):
        super(Graph_GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = Graph_Linear(num_nodes, input_size, 3 * hidden_size, bias=bias)
        self.h2h = Graph_Linear(num_nodes, hidden_size, 3 * hidden_size, bias=bias)
        self.reset_parameters()
    def reset_parameters(self):
        reset_parameters(self.named_parameters)
    def forward(self, x, hidden):
        # print("x: {}, hidden: {}".format(x.shape, hidden.shape))
        gate_x = self.x2h(x)
        gate_h = self.h2h(hidden)
        # print("gate_x: {}, gate_h: {}".format(gate_x.shape, gate_h.shape))
        # gate_x = gate_x.squeeze()
        # gate_h = gate_h.squeeze()
        # print("gate_x: {}, gate_h: {}".format(gate_x.shape, gate_h.shape))
        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)
        # print("i_r: {}, i_i: {}, i_n: {}".format(i_r.shape, i_i.shape, i_n.shape))
        # print("h_r: {}, h_i: {}, h_n: {}".format(h_r.shape, h_i.shape, h_n.shape))
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + (resetgate * h_n))
        # print("resetgate: {}, inputgate: {}, newgate: {}".format(resetgate.shape, inputgate.shape, newgate.shape))
        hy = newgate + inputgate * (hidden - newgate)
        # print("hy: {}".format(hy.shape))
        return hy

class Graph_GRUModel(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim, bias=True):
        super(Graph_GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru_cell = Graph_GRUCell(num_nodes, input_dim, hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        reset_parameters(self.named_parameters)

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = torch.zeros(x.size()[1], self.hidden_dim, device=x.device,dtype = x.dtype)
        for seq in range(x.size(0)):
            hidden = self.gru_cell(x[seq], hidden)
        return hidden

class AD_GAT_without_relational(nn.Module):
    def __init__(self, params, num_stock, shared_param=True, print_summary=False):
        super(AD_GAT_without_relational, self).__init__()
        self.params = params
        self.bn = nn.BatchNorm1d(self.params.feature_size*self.params.window_size, momentum=None)
        self.num_stock = num_stock
        self.shared_param = shared_param
        self.print_summary = print_summary
        if self.shared_param:
            self.GRUs_s = Graph_GRUModel(1, self.params.feature_size, self.params.lstm_num_units)
        else:
            self.GRUs_s = Graph_GRUModel(num_stock, self.params.feature_size, self.params.lstm_num_units)

        self.logit_f = nn.Linear(in_features=self.params.lstm_num_units, out_features=self.params.cum_labels + 1)

    def reset_parameters(self):
        reset_parameters(self.named_parameters)

    def forward(self, x_s):
        print("x_s: {}".format(x_s.shape))
        x_s = torch.permute(x_s, dims=(1,0,2))
        print("x_s: {}".format(x_s.shape))
        x_s = torch.reshape(x_s, (-1, self.params.window_size*self.params.feature_size))
        print("x_s: {}".format(x_s.shape))
        x_norm = self.bn(x_s)
        print("x_norm: {}".format(x_norm.shape))
        sentence = x_norm.reshape([-1, self.params.window_size, self.params.feature_size])
        print("sentence: {}".format(sentence.shape))
        if self.shared_param:
            if self.print_summary:
                print('shared param=True, sentence: {}'.format(sentence.shape))
                # summary(self.GRUs_s, input_data=sentence[0].unsqueeze(0), device=1) # set the device manually when debugging
                self.print_summary=False
            # x_s = torch.stack([self.GRUs_s(sentence[company_i].unsqueeze(0)) for company_i in range(sentence.size(0))])
            # x_s = x_s.squeeze(1)
        else:
            sentence = torch.permute(sentence, dims=(1,0,2))
            if self.print_summary:
                print('shared param=False, sentence: {}'.format(sentence.shape)) 
                # summary(self.GRUs_s, input_data=sentence, device=1) # set the device manually when debugging
                self.print_summary=False
            x_s = self.GRUs_s(sentence)
        # print("x_s: {}".format(x_s.shape))
        logits = self.logit_f(x_s.float())
        # print("logits: {}".format(logits.shape))
        logits = F.softmax(logits, dim=1)
        # print("logits: {}".format(logits.shape))
        logits = torch.cumsum(logits, dim=1)
        # print("logits: {}".format(logits.shape))
        eps = 5e-8
        logits = torch.clamp(logits, min=eps, max=1 - eps)
        # print("logits: {}".format(logits.shape))

        return logits

# print('tensor_input:{}'.format(tensor_input.shape))
# print('ts_input:{}'.format(ts_input.shape))

model = AD_GAT_without_relational(params, num_stock=NUM_STOCK, shared_param=False)
model.cuda(device=DEVICE)
model.to(torch.float)
optimizer = optim.Adam(model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)
summary(model, input_data=tensor_input)
logits = model(tensor_input)

# model_graph_gru = Graph_GRUModel(1, params.feature_size, params.lstm_num_units)
# model_graph_gru.cuda(device=DEVICE)
# model_graph_gru.to(torch.float)
# optimizer = optim.Adam(model_graph_gru.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)
# summary(model_graph_gru, input_data=ts_input)