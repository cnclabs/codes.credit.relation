import os
import pandas as pd

# Constants
DATA_DIR = "/home/cwlin/explainable_credit/data"
ALL_COMPANY_IDS_PATH = os.path.join(DATA_DIR, "edge_file", "all_company_ids.csv")
EDGE_FILE_DIR = os.path.join(DATA_DIR, "edge_file")
FEATURE_SIZE = 14
WINDOW_SIZE = 12
CUM_LABELS = 8

# Load all company ids
COMPANY_IDS = pd.read_csv(ALL_COMPANY_IDS_PATH, index_col=0).id.sort_values().unique()
NUM_COMPANIES = len(COMPANY_IDS) # 15786