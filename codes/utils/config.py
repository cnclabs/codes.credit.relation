import os
import pandas as pd

# Constants
DATA_DIR = "/home/cwlin/explainable_credit/data"
ALL_COMPANY_IDS_PATH = os.path.join(DATA_DIR, "edge_file", "all_company_ids.csv")
EDGE_FILE_DIR = os.path.join(DATA_DIR, "edge_file")

FEATURE_SIZE = 14   # each company is described by `d` unique features at each timestamp
WINDOW_SIZE = 12    # for each timestamp `t`, data from the previous `w` months, including the month `t` itself, are utilized as the input
CUM_LABELS = 8      # select `m` specific time points to discretize the support range (0, \infinty) for multiperiod default prediction

# Load all company ids
COMPANY_IDS = pd.read_csv(ALL_COMPANY_IDS_PATH, index_col=0).id.sort_values().unique()
NUM_COMPANIES = len(COMPANY_IDS) # 15786