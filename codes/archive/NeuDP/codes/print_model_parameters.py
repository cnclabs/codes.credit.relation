from main import *
import os
import pickle
import torch
from torchsummary import summary

data_dir = '/home/cwlin/explainable_credit/data'
cluster_setting = 'kmeans'
n_cluster = 100
inter_gat_hidn_dim=4
intra_gat_hidn_dim=4
lstm_hidn_dim=64
device = torch.device('cuda:0')

num_stock = len(pd.read_csv('/home/cwlin/explainable_credit/data/all_company_ids.csv', index_col=0).id.unique())
model = NeuDP(14, 12, 8+1, lstm_hidn_dim, num_stock, device)
model_path = f'/tmp2/cwlin/explainable_credit/codes.credit.relation.dev/NeuDP/experiments/index/NeuDP_12_index_lstm{lstm_hidn_dim}/last_weights/NeuDP_checkpoint.pt'

model.load_state_dict(torch.load(model_path))

model.to(device)

# test_input = torch.randn(12,num_stock,14)
# summary(model, input_size=test_input)

total_params = sum(p.numel() for p in model.parameters())

# Function to print model with hierarchy and parameters
def print_model_with_hierarchy(model, indent="\t"):
    total_params = 0
    for name, module in model.named_children():
        print(f"{indent}{name}:")
        if list(module.children()):
            child_params = print_model_with_hierarchy(module, indent + "  ")
            total_params += child_params
        else:
            for param_name, param in module.named_parameters():
                if param.requires_grad:
                    param_size = param.numel()
                    total_params += param_size
                    print(f"{indent}  {param_name}: Parameters: {param_size}")
    return total_params

# Print the model architecture with hierarchy and parameters
total_parameters = print_model_with_hierarchy(model)
print(f"Total Parameters: {total_parameters}")
