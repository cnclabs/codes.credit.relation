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
