############### Neural Default Prediction ###############
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

class Neural_Default_Prediction(nn.Module):
    def __init__(self, params):
        super(Neural_Default_Prediction, self).__init__()
        self.params = params
        
        # Batch Normalization Layer
        if self.params.batch_norm == "on":
            self.bn = nn.BatchNorm1d(self.params.feature_size*self.params.window_size, momentum=None)
        
        # ----- Models -----
        # MLP Model
        if 'mlp' in self.params.model_version:
            self.mlp_layer_1 = nn.Linear(in_features=self.params.feature_size*self.params.window_size, out_features=self.params.layer1_num_units)
            self.mlp_layer_2 = nn.Linear(in_features=self.params.layer1_num_units, out_features=self.params.layer2_num_units)
            self.logit_f = nn.Linear(in_features=self.params.layer2_num_units, out_features=self.params.cum_labels + 1)
        # RNNs
        else:
            # LSTM Model
            if 'lstm' in self.params.model_version:
                self.lstm = nn.LSTM(input_size=self.params.feature_size, hidden_size=self.params.lstm_num_units, batch_first=True)
                # With hidden layers
                if hasattr(self.params, 'hidden'):
                    self.logit_1 = nn.Linear(in_features=self.params.lstm_num_units, out_features=self.params.hidden)
                    self.logit_2 = nn.Linear(in_features=self.params.hidden, out_features=self.params.cum_labels + 1)
                else:
                    self.logit_f = nn.Linear(in_features=self.params.lstm_num_units, out_features=self.params.cum_labels + 1)
            # GRU Model
            else:
                self.gru = nn.GRU(input_size=self.params.feature_size, hidden_size=self.params.lstm_num_units, batch_first=True)
                self.logit_f = nn.Linear(in_features=self.params.lstm_num_units, out_features=self.params.cum_labels + 1)
    
    def forward(self, x):
        # Batch Normalization Layer
        if self.params.batch_norm == "on":
            x_norm = self.bn(x)
        else:
            x_norm = x
        
        # ----- Models -----
        # Reshape
        sentence = torch.reshape(x_norm, [-1, self.params.window_size, self.params.feature_size])
        
        # MLP Model
        if 'mlp' in self.params.model_version:
            logits = torch.sigmoid(self.mlp_layer_1(x_norm))
            logits = torch.sigmoid(self.mlp_layer_2(logits))
            logits = self.logit_f(logits)
        # RNNs
        else:
            # Use the hidden state of the last RNN unit -> hn
            # LSTM Model
            if 'lstm' in self.params.model_version:
                output, (hn, cn) = self.lstm(input=sentence)
            # GRU Model
            else:
                output, hn = self.gru(input=sentence)
        
        # ----- Output (After RNNs) -----
        if 'lstm' in self.params.model_version:
            # With hidden layers
            if hasattr(self.params, 'hidden'):
                logits = self.logit_1(hn)[0]
                logits = self.logit_2(logits)
            else:
                logits = self.logit_f(hn)[0]

        elif 'gru' in self.params.model_version:
            logits = self.logit_f(hn)[0]

        if 'type2' in self.params.model_version:
            logits = F.softmax(logits, dim=1)
            logits = torch.cumsum(logits, dim=1)
        else:
            logits = torch.sigmoid(logits)

        eps = 5e-8
        logits = torch.clamp(logits, min=eps, max=1 - eps)
        
        return logits

############### ADGAT without relational embedding ###############
from Layers import *

class Neural_Default_Prediction_revised(nn.Module):
    def __init__(self, params, num_stock):
        super(Neural_Default_Prediction_revised, self).__init__()
        self.params = params
        self.bn = nn.BatchNorm1d(self.params.feature_size*self.params.window_size, momentum=None)
        self.num_stock = num_stock
        self.gru = nn.GRU(input_size=self.params.feature_size, hidden_size=self.params.lstm_num_units, batch_first=True, dropout=params.dropout_rate)

        self.logit_f = nn.Linear(in_features=self.params.lstm_num_units, out_features=self.params.cum_labels + 1)

    def reset_parameters(self):
        reset_parameters(self.named_parameters)

    def forward(self, x_s):
        # 6,15786,14
        x_s = torch.permute(x_s, dims=(1,0,2))
        # 15786,6,14
        x_s = torch.reshape(x_s, (-1, self.params.window_size*self.params.feature_size))
        # 15786,6*14
        x_norm = self.bn(x_s)
        # 15786,6*14
        sentence = x_norm.reshape([-1, self.params.window_size, self.params.feature_size])
        # 15786,6,14
        # sentence = torch.permute(sentence, dims=(1,0,2))
        output, hn = self.gru(input=sentence)
        # # (15786,6,64), (1,15786,64)
        logits = self.logit_f(hn)[0]
        # 15786,9
        # x_s = self.GRUs_s(sentence)
        # logits = self.logit_f(x_s.float())
        logits = F.softmax(logits, dim=1)
        logits = torch.cumsum(logits, dim=1)
        eps = 5e-8
        logits = torch.clamp(logits, min=eps, max=1 - eps)

        return logits
        
############### ADGAT without relational embedding ###############
from Layers import *

class AD_GAT_without_relational(nn.Module):
    def __init__(self, args, params, num_stock, shared_param=False, print_summary=False, num_layers=1):
        super(AD_GAT_without_relational, self).__init__()
        self.args = args
        self.params = params
        self.bn = nn.BatchNorm1d(self.params.feature_size*self.params.window_size, momentum=None)
        self.num_stock = num_stock
        self.shared_param = shared_param
        self.print_summary = print_summary
        self.num_layers = num_layers
        if self.shared_param:
            if self.num_layers>1:
                # multilayer gru
                self.GRUs_s = Graph_GRUModel_multilayer(num_stock, self.params.feature_size, self.args.lstm_num_units, num_layers=2, dropout_rate=self.args.dropout_rate)
            else:
                #one-layer gru
                self.GRUs_s = Graph_GRUModel_shared(num_stock, self.params.feature_size, self.args.lstm_num_units)
        else:
            self.GRUs_s = Graph_GRUModel(num_stock, self.params.feature_size, self.args.lstm_num_units)

        self.logit_f = nn.Linear(in_features=self.args.lstm_num_units, out_features=self.params.cum_labels + 1)

        if self.print_summary:
            print(f'shared param={shared_param}')
            ts_input = torch.randn(self.params.window_size, num_stock, self.params.feature_size)
            summary(self.GRUs_s, input_data=ts_input, device=0)

    def reset_parameters(self):
        reset_parameters(self.named_parameters)

    def forward(self, x_s):
        # print("x_s: {}".format(x_s.shape))
        x_s = torch.permute(x_s, dims=(1,0,2))
        # print("x_s: {}".format(x_s.shape))
        x_s = torch.reshape(x_s, (-1, self.params.window_size*self.params.feature_size))
        # print("x_s: {}".format(x_s.shape))
        x_norm = self.bn(x_s)
        # print("x_norm: {}".format(x_norm.shape))
        sentence = x_norm.reshape([-1, self.params.window_size, self.params.feature_size])
        # print("sentence: {}".format(sentence.shape))
        # if self.print_summary:
        #     print('shared param=True, sentence: {}'.format(sentence.shape))
        #     # summary(self.GRUs_s, input_data=sentence[0].unsqueeze(0), device=1) # set the device manually when debugging
        #     self.print_summary=False

        sentence = torch.permute(sentence, dims=(1,0,2))
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

############### ADGAT ###############
from Layers import *

class AD_GAT(nn.Module):
    def __init__(self, params, num_stock, d_market, d_hidden, hidn_rnn, heads_att, hidn_att, dropout=0, alpha=0.2, infer = 1, relation_static = 0):
        super(AD_GAT, self).__init__()
        self.dropout = dropout
        self.params = params
        self.GRUs_s = Graph_GRUModel(num_stock, d_hidden, hidn_rnn)
        self.GRUs_r = Graph_GRUModel(num_stock, d_hidden, hidn_rnn)
        self.attentions = [
            Graph_Attention(hidn_rnn, hidn_att, dropout=dropout, alpha=alpha, residual=True, concat=True) for _
            in range(heads_att)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.logit_f = nn.Linear(in_features= hidn_rnn + heads_att*hidn_att, out_features=self.params.cum_labels + 1)

    def reset_parameters(self):
        reset_parameters(self.named_parameters)

    def get_relation(self,x_numerical, relation_static = None): 
        x_r = self.tensor(x_numerical)
        x_r = self.GRUs_r(x_r)
        relation = torch.stack([att.get_relation(x_r, relation_static=relation_static) for att in self.attentions])
        return relation

    def get_gate(self,x_numerical):
        x_s = self.tensor(x_numerical)
        x_s = self.GRUs_s(x_s)
        gate = torch.stack([att.get_gate(x_s) for att in self.attentions])
        return gate

    def forward(self, x_r, x_s, relation_static = None):
        x_r = self.GRUs_r(x_r)
        x_s = self.GRUs_s(x_s)
        x_r = F.dropout(x_r, self.dropout, training=self.training)
        x_s = F.dropout(x_s, self.dropout, training=self.training)
        ######### relational embeddings #############
        x = torch.cat([att(x_s, x_r, relation_static = relation_static) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([x, x_s], dim=1) # 15786, 360*2 ?

        logits = self.logit_f(x.float())
        logits = F.softmax(logits, dim=0)
        logits = torch.cumsum(logits, dim=0)
        eps = 5e-8
        logits = torch.clamp(logits, min=eps, max=1 - eps)
        #############################################
        return logits