import os 
import re
import json
import csv

ROOT = "/home/ybtu/codes.credit.relation.dev/NeuDP_GAT"

def initialize_report_file(report_file, headers):
    """Create the report file and write the header row."""
    try:
        with open(report_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    except IOError as e:
        print(f"Error initializing report file: {e}")

def read_json_file(file_path):
    """Read and return data from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except IOError as e:
        print(f"Error reading JSON file {file_path}: {e}")
        return None
    
def read_last_lines(file_path, line_count=8):
    """Read and return the last 'line_count' lines of a file."""
    try:
        with open(file_path, 'r') as file:
            return file.readlines()[-line_count:]
    except IOError as e:
        print(f"Error reading file {file_path}: {e}")
        return []

def compute_average(numbers):
    """Compute and return the average of a list of numbers."""
    return sum(numbers) / len(numbers) if numbers else 0

def extract_fold_number(dir_path):
    """Extract and return the fold number from the directory path."""
    path_components = dir_path.split(os.sep)
    fold_component = next((comp for comp in path_components if comp.startswith('fold_')), None)

    if fold_component:
        return fold_component.split('_')[-1] # Extract and return the fold number
    
    return None

def process_directory(dir_path, writer, experiment_type):
    """Process the specific directory."""
    # print(f"Processing {dir_path} for experiment type: {experiment_type}")

    fold_number = extract_fold_number(dir_path)

    args_file = os.path.join(dir_path, 'args.json') # Path to the args.json file
    args_data = read_json_file(args_file) # Read the args.json file
    if args_data:
        ar_file_path = os.path.join(dir_path, 'AR')
        rmsne_file_path = os.path.join(dir_path, 'RMSNE')

        ar_numbers = [float(line.strip()) for line in read_last_lines(ar_file_path)]
        rmsne_numbers = [float(line.strip()) for line in read_last_lines(rmsne_file_path)]

        avg_ar = compute_average(ar_numbers)
        avg_rmsne = compute_average(rmsne_numbers)

        writer.writerow([
            experiment_type, args_data.get('cluster_setting', ''), 
            args_data.get('n_cluster', ''), fold_number,
            args_data.get('num_epochs', ''), args_data.get('lstm_num_units', ''),
            args_data.get('intra_gat_hidn_dim', ''), args_data.get('inter_gat_hidn_dim', ''),
            args_data.get('learning_rate', ''), args_data.get('weight_decay', ''),
            *ar_numbers, avg_ar, 
            *rmsne_numbers, avg_rmsne
        ])


def find_matching_directories(base_dir, pattern, writer, experiment_type):
    """Find and process directories that match a given pattern."""
    pattern_regex = re.compile(pattern)
    for root, dirs, files in os.walk(base_dir):
        if pattern_regex.search(root):
            process_directory(root, writer, experiment_type)

if __name__ == "__main__":
    experiment_type = ["index", "time"]
    report_dir = os.path.join(ROOT, 'experiments', 'report.csv')
    
    # Define the headers for the report.csv
    headers = ["experiment_type", "cluster_setting", "n_cluster", "fold", "epoch", "lstm", "intra", "inter", "lr", "wd",
               "AR_01", "AR_02", "AR_03", "AR_04", "AR_05", "AR_06", "AR_07", "AR_08", "Avg_AR", 
               "RMSNE_01", "RMSNE_02", "RMSNE_03", "RMSNE_04", "RMSNE_05", "RMSNE_06", "RMSNE_07", "RMSNE_08", "Avg_RMSNE"]
    
    # Initialize the report.csv file
    initialize_report_file(report_dir, headers)

    pattern = r"NeuDP_GAT_\d+_(index|time)_lstm\d+_intra\d+_inter\d+_lr\d+\.\d+_wd\d+\.\d+$"
    with open(report_dir, 'a', newline='') as f:
        writer = csv.writer(f)
        for experiment_type in experiment_type:
            base_dir = os.path.join(ROOT, 'experiments', experiment_type)
            find_matching_directories(base_dir, pattern, writer, experiment_type)