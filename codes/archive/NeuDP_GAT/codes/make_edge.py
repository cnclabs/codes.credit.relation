import pickle
from itertools import combinations
import torch
import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--company_cluster', default='/home/ybtu/codes.credit.relation.dev/NeuDP_GAT/data/cluster_100/company_cluster_100.csv', type=str)
parser.add_argument('--company_ids', default='/home/ybtu/codes.credit.relation.dev/NeuDP_GAT/data/all_company_ids.csv', type=str)
parser.add_argument('--output_dir', default='/home/ybtu/codes.credit.relation.dev/NeuDP_GAT/data/cluster_100', type=str)

def generate_inner_edges(df: pd.DataFrame) -> list:
    """
    Generate inner edges from a dataframe containing company IDs and their clusters.
    
    Parameters:
    - df (pd.DataFrame): Dataframe with columns 'id' (representing company ID) and 'Cluster'.
    
    Returns:
    - list: List of tuples representing the inner edges between companies in the same cluster.
    """
    inner_edges = []

    # Group by 'Cluster' and create edges
    for _, group in df.groupby('Cluster'):
        companies = group['id'].tolist()
        for combo in combinations(companies, 2): # Get all pairs
            inner_edges.append(combo)   # (a, b)
            inner_edges.append(combo[::-1])  # (b, a)

    return inner_edges

def generate_outer_edges(df: pd.DataFrame) -> list:
    outer_edges = []

    # Extract unique sectors
    sectors = df['Cluster'].unique().tolist()
    print(f'num sectors: {len(sectors)}')
    for combo in combinations(sectors, 2): # Get all pairs
        outer_edges.append(combo)   # (a, b)
        outer_edges.append(combo[::-1])  # (b, a)
    
    return outer_edges

def map_companyID_to_index(company_id_list):
    """
    return a mapping dictionary from the original company IDs to the new indices
    Note: the order of the original data should be identical to company_id_list
    """
    id_to_index = {company_id: index for index, company_id in enumerate(company_id_list)}
    return id_to_index

def map_sector_to_index(sectors):
    sector_to_index = {sector: index for index, sector in enumerate(sectors)}
    return sector_to_index

def remap_edges(id_to_index, edges):
    remapped_edges = edges.clone()
    remapped_edges[0] = torch.tensor([id_to_index[original_id.item()] for original_id in edges[0]])
    remapped_edges[1] = torch.tensor([id_to_index[original_id.item()] for original_id in edges[1]])
    return remapped_edges

def remap_company_to_sector(company_to_sector, companyID_to_index, sector_to_index):
    remapped_dict = {}

    for company, sector in company_to_sector.items():
        company_idx = companyID_to_index.get(company)
        sector_idx = sector_to_index.get(sector)

        # Check if the company and sector have corresponding indices
        if company_idx is None:
            raise ValueError(f"No index found for company '{company}' in companyID_to_index.")
        if sector_idx is None:
            raise ValueError(f"No index found for sector '{sector}' in sector_to_index.")

        remapped_dict[company_idx] = sector_idx
        
    return remapped_dict

def main(args):

    company_cluster = pd.read_csv(args.company_cluster)
    company_ids = pd.read_csv(args.company_ids).id.tolist()

    sectors = list(sorted(company_cluster.Cluster.unique()))
    company_cluster_dict = dict()
    for id, cluster in zip(company_cluster.id, company_cluster.Cluster):
        company_cluster_dict[id] = cluster

    companyID_to_index = map_companyID_to_index(company_ids)
    sector_to_index = map_sector_to_index(sectors)

    inner_edge = generate_inner_edges(company_cluster)
    outer_edge = generate_outer_edges(company_cluster)

    inner_edge = torch.tensor(inner_edge, dtype=torch.int64).t() # (2, num_edges)
    outer_edge = torch.tensor(outer_edge, dtype=torch.int64).t() # (2, num_edges)

    print("Shape of inner_edge:", inner_edge.shape)
    print("Shape of outer_edge:", outer_edge.shape)

    torch.save(inner_edge, os.path.join(args.output_dir, 'inner_edge.pt'))
    torch.save(outer_edge, os.path.join(args.output_dir, 'outer_edge.pt'))

    inner_edge_idx = remap_edges(companyID_to_index, inner_edge)
    outer_edge_idx = remap_edges(sector_to_index, outer_edge)
    company_to_sector_idx = remap_company_to_sector(company_cluster_dict, companyID_to_index, sector_to_index)
    torch.save(inner_edge_idx, os.path.join(args.output_dir, 'inner_edge_idx.pt'))
    torch.save(outer_edge_idx, os.path.join(args.output_dir, 'outer_edge_idx.pt'))
    # torch.save(company_to_sector_idx, os.path.join(args.output_dir, 'company_to_sector_idx.pt'))
    with open(os.path.join(args.output_dir, 'company_to_sector_idx.pkl'), 'wb') as f:
        pickle.dump(company_to_sector_idx, f)


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)