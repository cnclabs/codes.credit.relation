
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from collections import Counter

from Layers import *

class NeuDP(nn.Module):
    def __init__(self, input_dim, window_size, output_dim, lstm_hidn_dim, num_company, device):
        super(NeuDP, self).__init__()

        # device
        self.device = device
        
        # basic parameters
        self.input_dim = input_dim      # feature_size
        self.window_size = window_size
        self.output_dim = output_dim    # cum_labels + 1
        self.lstm_hidn_dim = lstm_hidn_dim 
        self.num_company = num_company

        # hidden layers
        self.bn = nn.BatchNorm1d(input_dim * window_size, momentum=None) # Batch Normalization Layer
        # self.sequence_encoder = nn.GRU(input_size=input_dim, hidden_size=hidden_dim)
        self.sequence_encoder = Graph_GRUModel(self.num_company, input_dim, lstm_hidn_dim)

        # output layer
        self.logit_f = nn.Linear(in_features=lstm_hidn_dim, out_features=output_dim)

    def forward(self, daily_data_batch):
        # batch norm
        # print("Shape of daily_data_batch", daily_data_batch.shape)
        daily_data_batch = torch.permute(daily_data_batch, dims=(1, 0, 2))
        daily_data_batch = torch.reshape(daily_data_batch, (-1, self.window_size * self.input_dim))
        daily_data_batch_norm = self.bn(daily_data_batch)
        daily_data_batch_norm = daily_data_batch_norm.reshape([-1, self.window_size, self.input_dim])
        daily_data_batch_norm = torch.permute(daily_data_batch_norm, dims=(1, 0, 2))
        # print("Shape of daily_data_batch_norm", daily_data_batch_norm.shape)

        ## Compute the sequence embeddings
        sequence_embeddings = self.sequence_encoder(daily_data_batch_norm)
        # print("Shape of sequence_embeddings:", sequence_embeddings.shape)

        ## output
        logits = self.logit_f(sequence_embeddings.float())
        logits = F.softmax(logits, dim=1)
        logits = torch.cumsum(logits, dim=1)
        eps = 5e-8
        logits = torch.clamp(logits, min=eps, max=1 - eps)

        return logits

class CategoricalGraphAtt(nn.Module):
    def __init__(self, input_dim, window_size, output_dim, lstm_hidn_dim, intra_gat_hidn_dim, inter_gat_hidn_dim, inner_edge, outer_edge, company_to_sector, device):
        super(CategoricalGraphAtt, self).__init__()

        # device
        self.device = device
        
        # basic parameters
        self.input_dim = input_dim      # feature_size
        self.window_size = window_size
        self.output_dim = output_dim    # cum_labels + 1
        self.lstm_hidn_dim = lstm_hidn_dim 
        self.intra_gat_hidn_dim = intra_gat_hidn_dim 
        self.inter_gat_hidn_dim = inter_gat_hidn_dim 
        self.inner_edge = inner_edge    # expected dimension: (2, num_edges)
        self.outer_edge = outer_edge    # expected dimension: (2, num_edges)
        self.company_to_sector = company_to_sector # dictionary to map each company to its sector
        self.num_sector = len(set(company_to_sector.values()))
        self.num_company = len(set(company_to_sector.keys()))

        # hidden layers
        self.bn = nn.BatchNorm1d(input_dim * window_size, momentum=None) # Batch Normalization Layer
        # self.sequence_encoder = nn.GRU(input_size=input_dim, hidden_size=hidden_dim)
        self.sequence_encoder = Graph_GRUModel(self.num_company, input_dim, lstm_hidn_dim)
        self.intra_gat = GATConv(lstm_hidn_dim, intra_gat_hidn_dim) # GAT to conduct intra-sector relation
        self.cat_gat = GATConv(intra_gat_hidn_dim, inter_gat_hidn_dim) # GAT to conduct inter-sector relation
        self.fusion = nn.Linear(lstm_hidn_dim+intra_gat_hidn_dim+inter_gat_hidn_dim, lstm_hidn_dim)

        # output layer
        self.logit_f = nn.Linear(in_features=lstm_hidn_dim, out_features=output_dim)

    def sector_separater(self, full_embeddings):
        '''
        Separate the embeddings into different sectors based on a given mapping.

            Args:
            - full_embeddings (torch.Tensor): The embeddings tensor with shape (num_entities, hidden_dim) 
                                            representing all entities.

            Returns:
            - sectors_embeddings (dict): A dictionary where keys are sector identifiers and values are 
                                        tensors containing embeddings of all entities belonging to that sector.
        '''

        # explicit mapping to ensure each company is mapped to the correct sector
        sectors_embeddings = {} # Dictionary to store each company's inner_graph_embedding for each sector
        for company_idx, embedding in enumerate(full_embeddings):
            sector = self.company_to_sector[company_idx]

            if sector not in sectors_embeddings:
                sectors_embeddings[sector] = []

            sectors_embeddings[sector].append(embedding)
        
        # Convert lists to tensor
        for sector, embeddings in sectors_embeddings.items():
            sectors_embeddings[sector] = torch.stack(embeddings)
            # print(f"Shape of sector {sector}", sectors_embeddings[sector].shape)
            
        return sectors_embeddings

    def forward(self, daily_data_batch):
        # batch norm
        # print("Shape of daily_data_batch", daily_data_batch.shape)
        daily_data_batch = torch.permute(daily_data_batch, dims=(1, 0, 2))
        daily_data_batch = torch.reshape(daily_data_batch, (-1, self.window_size * self.input_dim))
        daily_data_batch_norm = self.bn(daily_data_batch)
        daily_data_batch_norm = daily_data_batch_norm.reshape([-1, self.window_size, self.input_dim])
        daily_data_batch_norm = torch.permute(daily_data_batch_norm, dims=(1, 0, 2))
        # print("Shape of daily_data_batch_norm", daily_data_batch_norm.shape)

        ## Compute the sequence embeddings
        sequence_embeddings = self.sequence_encoder(daily_data_batch_norm)
        # print("Shape of sequence_embeddings:", sequence_embeddings.shape)

        ## Conduct intra-sector embedding
        intra_sector_embeddings = self.intra_gat(sequence_embeddings, self.inner_edge)
        # print("Shape of intra_sector_embeddings:", intra_sector_embeddings.shape)

        ## MaxPool to get each sector's embedding
        sectors_embeddings = self.sector_separater(intra_sector_embeddings)
        for sector, embeddings in sectors_embeddings.items():
            sectors_embeddings[sector], _ = torch.max(sectors_embeddings[sector], dim=0) # simply adopt MaxPool on the sectors_embeddings of each sector

        sectors_embeddings = [sectors_embeddings[sector] for sector in sorted(sectors_embeddings.keys())]
        sectors_embeddings = torch.stack(sectors_embeddings, dim=0) # (num_sectors, hidden_dim)
        # print("Shape of sectors_embeddings:", sectors_embeddings.shape)
        
        ## Conduct inter-sector embedding
        sectors_embeddings = self.cat_gat(sectors_embeddings, self.outer_edge) # (num_sectors, hidden_dim)
        # print("Shape of sectors_embeddings:", sectors_embeddings.shape)

        ## fusion

        # duplicate sector embeddings for fusion
        sector_counts = Counter(self.company_to_sector.values())
        rep = torch.tensor([sector_counts[i] for i in sorted(sector_counts.keys())]).to(self.device) # Create repetitions tensor
        sectors_embeddings = torch.repeat_interleave(sectors_embeddings, rep, dim=0)
        # print("Shape of sectors_embeddings:", sectors_embeddings.shape)

        fusion_vec = torch.cat((sequence_embeddings, sectors_embeddings, intra_sector_embeddings), dim=-1)
        fusion_vec = torch.relu(self.fusion(fusion_vec))
        # print("Shape of fusion_vec", fusion_vec.shape)

        ## output

        logits = self.logit_f(fusion_vec.float())
        logits = F.softmax(logits, dim=1)
        logits = torch.cumsum(logits, dim=1)
        eps = 5e-8
        logits = torch.clamp(logits, min=eps, max=1 - eps)

        return logits
