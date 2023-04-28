############### Neural Default Prediction ###############
import torch
import torch.nn as nn
import torch.nn.functional as F
from Layers import *

# Unified ADGAT with without relation
class Neural_Default_Prediction_revised(nn.Module):
    def __init__(self, params, adgat_relation):
        super(Neural_Default_Prediction_revised, self).__init__()
        self.params = params
        self.adgat_relation = adgat_relation

        # Batch Normalization Layer
        self.bn = nn.BatchNorm1d(self.params.feature_size*self.params.window_size, momentum=None)

        # If shared parameter gru model
        if params.shared_parameter:
            if self.params.number_of_layer>1:
                self.GRUs_s = Graph_GRUModel_shared_multilayer(num_nodes=self.params.num_stock, input_dim=self.params.feature_size, hidden_dim=self.params.lstm_num_units, num_layers=self.params.number_of_layer, dropout_rate=self.params.dropout_rate)
            else:
                self.GRUs_s = Graph_GRUModel_shared(num_nodes=self.params.num_stock, input_dim=self.params.feature_size, hidden_dim=self.params.lstm_num_units)
        else:
            if self.params.number_of_layer>1:
                self.GRUs_s = Graph_GRUModel(num_nodes=self.params.num_stock, input_dim=self.params.feature_size, hidden_dim=self.params.lstm_num_units)
            else:
                self.GRUs_s = Graph_GRUModel_multilayer(num_nodes=self.params.num_stock, input_dim=self.params.feature_size, hidden_dim=self.params.lstm_num_units, num_layers=self.params.number_of_layer, dropout_rate=self.params.dropout_rate)

        # ADGAT relation model
        if self.adgat_relation:
            self.logit_f = nn.Linear(in_features= self.params.lstm_num_units + self.params.heads_att * self.params.hidn_att, out_features=self.params.cum_labels + 1)
            
            # GRUs_r
            if params.shared_parameter:
                if self.params.number_of_layer>1:
                    self.GRUs_r = Graph_GRUModel_shared_multilayer(num_nodes=self.params.num_stock, input_dim=self.params.feature_size, hidden_dim=self.params.lstm_num_units, num_layers=self.params.number_of_layer, dropout_rate=self.params.dropout_rate)
                else:
                    self.GRUs_r = Graph_GRUModel_shared(num_nodes=self.params.num_stock, input_dim=self.params.feature_size, hidden_dim=self.params.lstm_num_units)
            else:
                if self.params.number_of_layer>1:
                    self.GRUs_r = Graph_GRUModel(num_nodes=self.params.num_stock, input_dim=self.params.feature_size, hidden_dim=self.params.lstm_num_units)
                else:
                    self.GRUs_r = Graph_GRUModel_multilayer(num_nodes=self.params.num_stock, input_dim=self.params.feature_size, hidden_dim=self.params.lstm_num_units, num_layers=self.params.number_of_layer, dropout_rate=self.params.dropout_rate)
            # attention
            self.attentions = [
                Graph_Attention(self.params.lstm_num_units, self.params.hidn_att, num_stock=self.params.num_stock, dropout=self.params.dropout_rate, alpha=self.params.alpha, residual=True, concat=True) for _
                in range(self.params.heads_att)]
            for i, attention in enumerate(self.attentions):
                self.add_module('attention_{}'.format(i), attention)
        else:
            self.logit_f = nn.Linear(in_features= self.params.lstm_num_units, out_features=self.params.cum_labels + 1)

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
        x_s = x_s.permute(1, 0, 2)
        x_s = torch.reshape(x_s, (-1, self.params.window_size*self.params.feature_size))
        x_norm = self.bn(x_s)
        
        # ----- Models -----
        # Reshape
        sentence = x_norm.reshape([-1, self.params.window_size, self.params.feature_size])

        sentence = sentence.permute(1, 0, 2)
        x_s = self.GRUs_s(sentence)
        if self.adgat_relation:
            x_r = self.GRUs_r(sentence)
            x_r = F.dropout(x_r, self.params.dropout_rate, training=self.training)
            x_s = F.dropout(x_s, self.params.dropout_rate, training=self.training)
            
            x = torch.cat([att(x_s, x_r, relation_static = relation_static) for att in self.attentions], dim=1)
            x = F.dropout(x, self.params.dropout_rate, training=self.training)
            x_s = torch.cat([x, x_s], dim=1)       
        
        logits = self.logit_f(x_s.float())
        logits = F.softmax(logits, dim=1)
        logits = torch.cumsum(logits, dim=1)
        eps = 5e-8
        logits = torch.clamp(logits, min=eps, max=1 - eps)

        return logits