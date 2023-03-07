############### Neural Default Prediction ###############
import torch
import torch.nn as nn
import torch.nn.functional as F

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

class AD_GAT_without_relational(nn.Module):
    def __init__(self, params, num_stock):
        super(AD_GAT_without_relational, self).__init__()
        self.params = params
        self.bn = nn.BatchNorm1d(self.params.feature_size*self.params.window_size, momentum=None)
        self.num_stock = num_stock
        # self.gru = nn.GRU(input_size=self.params.feature_size, hidden_size=self.params.lstm_num_units, batch_first=True, dropout=params.dropout_rate)
        self.GRUs_s = Graph_GRUModel(num_stock, self.params.feature_size, self.params.lstm_num_units)

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
        sentence = torch.permute(sentence, dims=(1,0,2))
        # output, hn = self.gru(input=sentence)
        # # (15786,6,64), (1,15786,64)
        # logits = self.logit_f(hn)[0]
        # 15786,9
        x_s = self.GRUs_s(sentence)
        logits = self.logit_f(x_s.float())
        logits = F.softmax(logits, dim=1)
        logits = torch.cumsum(logits, dim=1)
        eps = 5e-8
        logits = torch.clamp(logits, min=eps, max=1 - eps)

        return logits

############### ADGAT ###############
from Layers import *

class AD_GAT(nn.Module):
    def __init__(self, params, num_stock, d_hidden, hidn_rnn, heads_att, hidn_att, dropout=0, alpha=0.2, infer = 1, relation_static = 0):
        super(AD_GAT, self).__init__()
        self.dropout = dropout
        self.params = params
        self.num_stock = num_stock

        # Batch Normalization Layer
        if self.params.batch_norm == "on":
            self.bn = nn.BatchNorm1d(self.params.feature_size*self.params.window_size, momentum=None)

        # MLP Model
        if 'mlp' in self.params.model_version:
            self.mlp_layer_1_s = nn.Linear(in_features=self.params.feature_size*self.params.window_size, out_features=self.params.layer1_num_units)
            self.mlp_layer_2_s = nn.Linear(in_features=self.params.layer1_num_units, out_features=self.params.layer2_num_units)

            self.mlp_layer_1_r = nn.Linear(in_features=self.params.feature_size*self.params.window_size, out_features=self.params.layer1_num_units)
            self.mlp_layer_2_r = nn.Linear(in_features=self.params.layer1_num_units, out_features=self.params.layer2_num_units)
            self.logit_f = nn.Linear(in_features= self.params.layer2_num_units + heads_att * hidn_att, out_features=self.params.cum_labels + 1)

            self.attentions = [
                Graph_Attention(self.params.layer2_num_units, hidn_att, dropout=dropout, alpha=alpha, residual=True, concat=True) for _
                in range(heads_att)]
            
        # RNNs
        else:
            # LSTM Model
            if 'lstm' in self.params.model_version:
                self.lstm_r = nn.LSTM(input_size=self.params.feature_size, hidden_size=self.params.lstm_num_units, batch_first=True)
                self.lstm_s = nn.LSTM(input_size=self.params.feature_size, hidden_size=self.params.lstm_num_units, batch_first=True)
                self.attentions = [
                    Graph_Attention(self.params.lstm_num_units, hidn_att, dropout=dropout, alpha=alpha, residual=True, concat=True) for _
                    in range(heads_att)]
                # With hidden layers
                if hasattr(self.params, 'hidden'):
                    self.logit_1 = nn.Linear(in_features=self.params.lstm_num_units + heads_att * hidn_att, out_features=self.params.hidden)
                    self.logit_2 = nn.Linear(in_features=self.params.hidden, out_features=self.params.cum_labels + 1)
                else:
                    self.logit_f = nn.Linear(in_features=self.params.lstm_num_units + heads_att * hidn_att, out_features=self.params.cum_labels + 1)
            # GRU Model (ADGAT)
            else:
                self.gru_s = nn.GRU(input_size=self.params.feature_size, hidden_size=self.params.lstm_num_units, batch_first=True)
                self.gru_r = nn.GRU(input_size=self.params.feature_size, hidden_size=self.params.lstm_num_units, batch_first=True)
                # self.GRUs_s = Graph_GRUModel(num_stock, d_hidden, hidn_rnn)
                # self.GRUs_r = Graph_GRUModel(num_stock, d_hidden, hidn_rnn)
                self.logit_f = nn.Linear(in_features= hidn_rnn + heads_att * hidn_att, out_features=self.params.cum_labels + 1)

                self.attentions = [
                    Graph_Attention(hidn_rnn, hidn_att, num_stock=num_stock, dropout=dropout, alpha=alpha, residual=True, concat=True) for _
                    in range(heads_att)]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

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

    def forward(self, x_r, relation_static = None):
        # Batch Normalization Layer
        if self.params.batch_norm == "on":
            x_r = torch.permute(x_r, (1, 0, 2))
            x_r = torch.reshape(x_r, (-1, self.params.window_size * self.params.feature_size))
            x_r = self.bn(x_r)
        else:
            x_r = x_r
        
        # ----- Models -----
        # Reshape
        x_r = torch.reshape(x_r, [-1, self.params.window_size, self.params.feature_size])

        # MLP Model
        if 'mlp' in self.params.model_version:
            # x_s = x_r.permute(1, 0, 2)
            # x_r = x_r.permute(1, 0, 2)
            x_r = torch.reshape(x_r, (-1, self.params.feature_size*self.params.window_size))
            x_s = x_r
            # x_s = torch.reshape(x_s, (-1, self.params.feature_size*self.params.window_size))
            x_r = torch.sigmoid(self.mlp_layer_1_r(x_r))
            x_s = torch.sigmoid(self.mlp_layer_1_s(x_s))
            x_r = torch.sigmoid(self.mlp_layer_2_r(x_r))
            x_s = torch.sigmoid(self.mlp_layer_2_s(x_s))

        # RNNs
        else:
            # Use the hidden state of the last RNN unit -> hn
            # LSTM Model
            if 'lstm' in self.params.model_version:
                # print("error test")
                output_r, (x_s, cn_r) = self.lstm_s(input=x_r)
                output_s, (x_r, cn_s) = self.lstm_r(input=x_r)
                x_s = torch.squeeze(x_s)
                x_r = torch.squeeze(x_r)
            # GRU Model
            else:
                output, x_s = self.gru_s(input=x_r)
                output, x_r = self.gru_r(input=x_r)
                x_s = torch.squeeze(x_s)
                x_r = torch.squeeze(x_r)
                # x_s = self.GRUs_s(x_r)
                # x_r = self.GRUs_r(x_r)
                # x_r = F.dropout(x_r, self.dropout, training=self.training)
                # x_s = F.dropout(x_s, self.dropout, training=self.training)
        
        # ----- Output (After RNNs) -----

        # ADGAT
        x = torch.cat([att(x_s, x_r, relation_static = relation_static) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)

        x = torch.cat([x, x_s], dim=1) # 15786, 360*2 ?

        if 'mlp' in self.params.model_version:
            logits = self.logit_f(x.float())
            # logits = self.logit_f(logits)

        elif 'lstm' in self.params.model_version:
            # With hidden layers
            if hasattr(self.params, 'hidden'):
                logits = self.logit_1(x.float())
                logits = self.logit_2(logits)
            else:
                logits = self.logit_f(x.float())

        elif 'gru' in self.params.model_version:
            logits = self.logit_f(x.float())
            # logits = self.logit_f(logits)


        if 'type2' in self.params.model_version:
            logits = F.softmax(logits, dim=1)
            logits = torch.cumsum(logits, dim=1)
        else:
            logits = torch.sigmoid(logits)

        eps = 5e-8
        logits = torch.clamp(logits, min=eps, max=1 - eps)
        
        return logits

########################################################################################
# Unified ADGAT with without relation
# class Neural_Default_Prediction_revised(nn.Module):
#     def __init__(self, params, adgat_relation):
#         super(Neural_Default_Prediction_revised, self).__init__()
#         self.params = params
#         self.adgat_relation = adgat_relation

#         """
#             {
#                 For model parameter:
#                 {
#                     # In json
#                     "feature_size": 14,
#                     "window_size": 1,
#                     "model_version": "gru-type2",
#                     "layer_norm": "off",
#                     "batch_norm": "on",
#                     "label_type": "cum",
#                     "cum_labels": 8,
#                     "lstm_num_units": 64, == "hidden_size": 64
#                     "buffer_size": 1e6,

#                     # Not in json
#                     "relation_static": None,
#                     "adgat_relation": ,
#                     "gru_model": "GRU, Graph_GRU, Graph_GRU_shared",
#                     "number_of_layer": 1,
#                     "heads_att": 6,
#                     "hidn_att": 60
#                 }
#                 For hyperparameter:
#                 {
#                     # In json
#                     "learning_rate": 1e-4,
#                     "batch_size": 256,
#                     "num_epochs": 20,
#                     "dropout_rate": 0.5,
#                     "weight_decay": 1e-5,
#                     "save_summary_steps": 100

#                     # Not in json
#                     "patience": 20,
#                     "num_epochs": 300,
#                     "alpha": 0.2
#                 }
#             }
#         """


#         # Batch Normalization Layer
#         if self.params.batch_norm == "on":
#             self.bn = nn.BatchNorm1d(self.params.feature_size*self.params.window_size, momentum=None)

#         # MLP Model
#         if 'mlp' in self.params.model_version:
#             self.mlp_layer_1_s = nn.Linear(in_features=self.params.feature_size*self.params.window_size, out_features=self.params.layer1_num_units)
#             self.mlp_layer_2_s = nn.Linear(in_features=self.params.layer1_num_units, out_features=self.params.layer2_num_units)

#             if self.adgat_relation:
#                 self.mlp_layer_1_r = nn.Linear(in_features=self.params.feature_size*self.params.window_size, out_features=self.params.layer1_num_units)
#                 self.mlp_layer_2_r = nn.Linear(in_features=self.params.layer1_num_units, out_features=self.params.layer2_num_units)
#                 self.logit_f = nn.Linear(in_features= self.params.layer2_num_units + self.params.heads_att * self.params.hidn_att, out_features=self.params.cum_labels + 1)
#                 self.attentions = [
#                     Graph_Attention(self.params.layer2_num_units, self.params.hidn_att, dropout=self.params.dropout_rate, alpha=self.params.alpha, residual=True, concat=True) for _
#                     in range(self.params.heads_att)]
#             else:
#                 self.logit_f = nn.Linear(in_features= self.params.layer2_num_units, out_features=self.params.cum_labels + 1)
            
#         # RNNs
#         else:
#             # LSTM Model
#             if 'lstm' in self.params.model_version:
#                 self.lstm_s = nn.LSTM(input_size=self.params.feature_size, hidden_size=self.params.lstm_num_units, batch_first=True)
#                 if self.adgat_relation:
#                     self.lstm_r = nn.LSTM(input_size=self.params.feature_size, hidden_size=self.params.lstm_num_units, batch_first=True)
#                     self.attentions = [
#                         Graph_Attention(self.params.lstm_num_units, self.params.hidn_att, dropout=self.params.dropout_rate, alpha=self.params.alpha, residual=True, concat=True) for _
#                         in range(self.params.heads_att)]
#                     # With hidden layers
#                     if hasattr(self.params, 'hidden'):
#                         self.logit_1 = nn.Linear(in_features=self.params.lstm_num_units + self.params.heads_att * self.params.hidn_att, out_features=self.params.hidden)
#                         self.logit_2 = nn.Linear(in_features=self.params.hidden, out_features=self.params.cum_labels + 1)
#                     else:
#                         self.logit_f = nn.Linear(in_features=self.params.lstm_num_units + self.params.heads_att * self.params.hidn_att, out_features=self.params.cum_labels + 1)
#                 else:
#                     # With hidden layers
#                     if hasattr(self.params, 'hidden'):
#                         self.logit_1 = nn.Linear(in_features=self.params.lstm_num_units, out_features=self.params.hidden)
#                         self.logit_2 = nn.Linear(in_features=self.params.hidden, out_features=self.params.cum_labels + 1)
#                     else:
#                         self.logit_f = nn.Linear(in_features=self.params.lstm_num_units, out_features=self.params.cum_labels + 1)

#             # GRU Model (ADGAT)
#             else:
#                 if self.adgat_relation:
#                     if params.gru_model == 'GRU':
#                         self.gru_s = nn.GRU(input_size=self.params.feature_size, hidden_size=self.params.lstm_num_units, batch_first=False, dropout=params.dropout_rate)
#                         self.gru_r = nn.GRU(input_size=self.params.feature_size, hidden_size=self.params.lstm_num_units, batch_first=False, dropout=params.dropout_rate)
#                     if params.gru_model == 'Graph_GRU':
#                         self.GRUs_s = Graph_GRUModel(num_nodes=self.params.num_stock, input_dim=self.params.feature_size, hidden_dim=self.params.lstm_num_units)
#                         self.GRUs_r = Graph_GRUModel(num_nodes=self.params.num_stock, input_dim=self.params.feature_size, hidden_dim=self.params.lstm_num_units)
#                     elif params.gru_model == 'Graph_GRU_shared':
#                         self.GRUs_s = Graph_GRUModel_shared(num_nodes=self.params.num_stock, input_dim=self.params.feature_size, hidden_dim=self.params.lstm_num_units)
#                         self.GRUs_r = Graph_GRUModel_shared(num_nodes=self.params.num_stock, input_dim=self.params.feature_size, hidden_dim=self.params.lstm_num_units)
#                     self.logit_f = nn.Linear(in_features= self.params.lstm_num_units + self.params.heads_att * self.params.hidn_att, out_features=self.params.cum_labels + 1)
#                     self.attentions = [
#                         Graph_Attention(self.params.lstm_num_units, self.params.hidn_att, num_stock=self.params.num_stock, dropout=self.params.dropout_rate, alpha=self.params.alpha, residual=True, concat=True) for _
#                         in range(self.params.heads_att)]
#                 else:
#                     if params.gru_model == 'GRU':
#                         self.gru_s = nn.GRU(input_size=self.params.feature_size, hidden_size=self.params.lstm_num_units, batch_first=False, dropout=params.dropout_rate)
#                     if params.gru_model == 'Graph_GRU':
#                         self.GRUs_s = Graph_GRUModel(num_nodes=self.params.num_stock, input_dim=self.params.feature_size, hidden_dim=self.params.lstm_num_units)
#                     elif params.gru_model == 'Graph_GRU_shared':
#                         self.GRUs_s = Graph_GRUModel_shared(num_nodes=self.params.num_stock, input_dim=self.params.feature_size, hidden_dim=self.params.lstm_num_units)
#                     self.logit_f = nn.Linear(in_features= self.params.lstm_num_units, out_features=self.params.cum_labels + 1)

#         if self.adgat_relation:
#             for i, attention in enumerate(self.attentions):
#                 self.add_module('attention_{}'.format(i), attention)

#     def reset_parameters(self):
#         reset_parameters(self.named_parameters)

#     def get_relation(self,x_numerical, relation_static = None): 
#         x_r = self.tensor(x_numerical)
#         x_r = self.GRUs_r(x_r)
#         relation = torch.stack([att.get_relation(x_r, relation_static=relation_static) for att in self.attentions])
#         return relation

#     def get_gate(self,x_numerical):
#         x_s = self.tensor(x_numerical)
#         x_s = self.GRUs_s(x_s)
#         gate = torch.stack([att.get_gate(x_s) for att in self.attentions])
#         return gate

#     def forward(self, x_s, relation_static = None):
#         # Batch Normalization Layer
#         if self.params.batch_norm == "on":
#             x_s = torch.permute(x_s, (1, 0, 2))
#             x_s = torch.reshape(x_s, (-1, self.params.window_size * self.params.feature_size))
#             x_s = self.bn(x_s)
#         else:
#             x_s = x_s
        
#         # ----- Models -----
#         # Reshape
#         x_s = torch.reshape(x_s, [-1, self.params.window_size, self.params.feature_size])
#         x_s = torch.permute(x_s, (1, 0, 2))

#         # MLP Model
#         if 'mlp' in self.params.model_version:
            
#             x_s = torch.reshape(x_s, (-1, self.params.feature_size*self.params.window_size))

#             if self.adgat_relation:
#                 x_r = x_s
#                 x_r = torch.sigmoid(self.mlp_layer_1_r(x_r))
#                 x_s = torch.sigmoid(self.mlp_layer_1_s(x_s))
#                 x_r = torch.sigmoid(self.mlp_layer_2_r(x_r))
#                 x_s = torch.sigmoid(self.mlp_layer_2_s(x_s))
#             else:
#                 x_s = torch.sigmoid(self.mlp_layer_1_s(x_s))
#                 x_s = torch.sigmoid(self.mlp_layer_2_s(x_s))

#         # RNNs
#         else:
#             # Use the hidden state of the last RNN unit -> hn
#             # LSTM Model
#             if 'lstm' in self.params.model_version:
#                 if self.adgat_relation:
#                     x_r = x_s
#                     output_r, (x_r, cn_r) = self.lstm_r(input=x_r)
#                     output_s, (x_s, cn_s) = self.lstm_s(input=x_s)
#                     x_s = torch.squeeze(x_s)
#                     x_r = torch.squeeze(x_r)
#                 else:
#                     output_s, (x_s, cn_s) = self.lstm_s(input=x_s)

#             # GRU Model
#             else:
#                 if self.adgat_relation:
#                     x_r = x_s
#                     if self.params.gru_model == 'GRU':
#                         output, x_s = self.gru_s(input=x_s)
#                         output, x_r = self.gru_r(input=x_r)
#                         x_s = torch.squeeze(x_s)
#                         x_r = torch.squeeze(x_r)
#                     else:
#                         x_s = self.GRUs_s(x_s)
#                         x_r = self.GRUs_r(x_r)
#                         x_r = F.dropout(x_r, self.params.dropout_rate, training=self.training)
#                         x_s = F.dropout(x_s, self.params.dropout_rate, training=self.training)
#                 else:
#                     if self.params.gru_model == 'GRU':
#                         output, x_s = self.gru_s(input=x_s)
#                         x_s = torch.squeeze(x_s)
#                     else:
#                         x_s = self.GRUs_s(x_s)
#                         x_s = F.dropout(x_s, self.params.dropout_rate, training=self.training)
        
#         # ----- Output (After RNNs) -----

#         # ADGAT
#         if self.adgat_relation:
#             x = torch.cat([att(x_s, x_r, relation_static = self.params.relation_static) for att in self.attentions], dim=1)
#             x = F.dropout(x, self.params.dropout_rate, training=self.training)

#             x_s = torch.cat([x, x_s], dim=1) # 15786, 360*2 ?

#         if 'mlp' in self.params.model_version:
#             logits = self.logit_f(x_s.float())

#         elif 'lstm' in self.params.model_version:
#             # With hidden layers
#             if hasattr(self.params, 'hidden'):
#                 logits = self.logit_1(x_s.float())
#                 logits = self.logit_2(logits)
#             else:
#                 logits = self.logit_f(x_s.float())

#         elif 'gru' in self.params.model_version:
#             logits = self.logit_f(x_s.float())

#         if 'type2' in self.params.model_version:
#             logits = F.softmax(logits, dim=1)
#             logits = torch.cumsum(logits, dim=1)
#         else:
#             logits = torch.sigmoid(logits)

#         eps = 5e-8
#         logits = torch.clamp(logits, min=eps, max=1 - eps)
        
#         return logits
    

########################################################################################
# Unified ADGAT with without relation
class Neural_Default_Prediction_revised(nn.Module):
    def __init__(self, params, adgat_relation):
        super(Neural_Default_Prediction_revised, self).__init__()
        self.params = params
        self.adgat_relation = adgat_relation

        # Batch Normalization Layer
        if self.params.batch_norm == "on":
            self.bn = nn.BatchNorm1d(self.params.feature_size*self.params.window_size, momentum=None)

        # GRU Model (ADGAT)
        if self.adgat_relation:
            if params.gru_model == 'GRU':
                self.gru_s = nn.GRU(input_size=self.params.feature_size, hidden_size=self.params.lstm_num_units, num_layers=self.params.number_of_layer, batch_first=False, dropout=self.params.dropout_rate)
                self.gru_r = nn.GRU(input_size=self.params.feature_size, hidden_size=self.params.lstm_num_units, num_layers=self.params.number_of_layer,batch_first=False, dropout=self.params.dropout_rate)
            if params.gru_model == 'Graph_GRU':
                if self.params.number_of_layer>1:
                    # multilayer gru
                    self.GRUs_s = Graph_GRUModel_multilayer(num_nodes=self.params.num_stock, input_dim=self.params.feature_size, hidden_dim=self.params.lstm_num_units, num_layers=self.params.number_of_layer, dropout_rate=self.params.dropout_rate)
                    self.GRUs_r = Graph_GRUModel_multilayer(num_nodes=self.params.num_stock, input_dim=self.params.feature_size, hidden_dim=self.params.lstm_num_units, num_layers=self.params.number_of_layer, dropout_rate=self.params.dropout_rate)
                else:
                    self.GRUs_s = Graph_GRUModel(num_nodes=self.params.num_stock, input_dim=self.params.feature_size, hidden_dim=self.params.lstm_num_units)
                    self.GRUs_r = Graph_GRUModel(num_nodes=self.params.num_stock, input_dim=self.params.feature_size, hidden_dim=self.params.lstm_num_units)
            elif params.gru_model == 'Graph_GRU_shared':
                if self.params.number_of_layer>1:
                    # multilayer gru
                    self.GRUs_s = Graph_GRUModel_shared_multilayer(num_nodes=self.params.num_stock, input_dim=self.params.feature_size, hidden_dim=self.params.lstm_num_units, num_layers=self.params.number_of_layer, dropout_rate=self.params.dropout_rate)
                    self.GRUs_r = Graph_GRUModel_shared_multilayer(num_nodes=self.params.num_stock, input_dim=self.params.feature_size, hidden_dim=self.params.lstm_num_units, num_layers=self.params.number_of_layer, dropout_rate=self.params.dropout_rate)
                else:
                    self.GRUs_s = Graph_GRUModel_shared(num_nodes=self.params.num_stock, input_dim=self.params.feature_size, hidden_dim=self.params.lstm_num_units)
                    self.GRUs_r = Graph_GRUModel_shared(num_nodes=self.params.num_stock, input_dim=self.params.feature_size, hidden_dim=self.params.lstm_num_units)
            self.logit_f = nn.Linear(in_features= self.params.lstm_num_units + self.params.heads_att * self.params.hidn_att, out_features=self.params.cum_labels + 1)
            self.attentions = [
                Graph_Attention(self.params.lstm_num_units, self.params.hidn_att, num_stock=self.params.num_stock, dropout=self.params.dropout_rate, alpha=self.params.alpha, residual=True, concat=True) for _
                in range(self.params.heads_att)]
        else:
            if params.gru_model == 'GRU':
                self.gru_s = nn.GRU(input_size=self.params.feature_size, hidden_size=self.params.lstm_num_units, num_layers=self.params.number_of_layer, batch_first=False, dropout=self.params.dropout_rate)
            if params.gru_model == 'Graph_GRU':
                if self.params.number_of_layer>1:
                    # multilayer gru
                    self.GRUs_s = Graph_GRUModel_multilayer(num_nodes=self.params.num_stock, input_dim=self.params.feature_size, hidden_dim=self.params.lstm_num_units, num_layers=self.params.number_of_layer, dropout_rate=self.params.dropout_rate)
                else:
                    self.GRUs_s = Graph_GRUModel(num_nodes=self.params.num_stock, input_dim=self.params.feature_size, hidden_dim=self.params.lstm_num_units)
            elif params.gru_model == 'Graph_GRU_shared':
                if self.params.number_of_layer>1:
                    # multilayer gru
                    self.GRUs_s = Graph_GRUModel_shared_multilayer(num_nodes=self.params.num_stock, input_dim=self.params.feature_size, hidden_dim=self.params.lstm_num_units, num_layers=self.params.number_of_layer, dropout_rate=self.params.dropout_rate)
                else:
                    self.GRUs_s = Graph_GRUModel_shared(num_nodes=self.params.num_stock, input_dim=self.params.feature_size, hidden_dim=self.params.lstm_num_units)
            self.logit_f = nn.Linear(in_features= self.params.lstm_num_units, out_features=self.params.cum_labels + 1)

        if self.adgat_relation:
            for i, attention in enumerate(self.attentions):
                self.add_module('attention_{}'.format(i), attention)

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

    def forward(self, x_s, relation_static = None):
        # Batch Normalization Layer
        if self.params.batch_norm == "on":
            x_s = torch.permute(x_s, (1, 0, 2))
            x_s = torch.reshape(x_s, (-1, self.params.window_size * self.params.feature_size))
            x_s = self.bn(x_s)
        else:
            x_s = x_s
        
        # ----- Models -----
        # Reshape
        x_s = torch.reshape(x_s, [-1, self.params.window_size, self.params.feature_size])
        x_s = torch.permute(x_s, (1, 0, 2))

        # GRU Model
        if self.adgat_relation:
            x_r = x_s
            if self.params.gru_model == 'GRU':
                output, x_s = self.gru_s(input=x_s)
                output, x_r = self.gru_r(input=x_r)
                x_s = torch.squeeze(x_s)
                x_r = torch.squeeze(x_r)
            else:
                x_s = self.GRUs_s(x_s)
                x_r = self.GRUs_r(x_r)
                x_r = F.dropout(x_r, self.params.dropout_rate, training=self.training)
                x_s = F.dropout(x_s, self.params.dropout_rate, training=self.training)
        else:
            if self.params.gru_model == 'GRU':
                output, x_s = self.gru_s(input=x_s)
                x_s = torch.squeeze(x_s)
            else:
                x_s = self.GRUs_s(x_s)
                x_s = F.dropout(x_s, self.params.dropout_rate, training=self.training)
        
        # ----- Output (After RNNs) -----

        # ADGAT
        if self.adgat_relation:
            x = torch.cat([att(x_s, x_r, relation_static = self.params.relation_static) for att in self.attentions], dim=1)
            x = F.dropout(x, self.params.dropout_rate, training=self.training)

            x_s = torch.cat([x, x_s], dim=1) # 15786, 360*2 ?

        logits = self.logit_f(x_s.float())
        logits = F.softmax(logits, dim=1)
        logits = torch.cumsum(logits, dim=1)
        eps = 5e-8
        logits = torch.clamp(logits, min=eps, max=1 - eps)
        
        return logits