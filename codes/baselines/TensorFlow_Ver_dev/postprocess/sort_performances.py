import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--json_file', help="json file to sort, e.g. valid_metrics_pred_best_weights.json")
parser.add_argument('--sort_by', default='AR', help="sort by which metric, AR or RMSNE")
parser.add_argument('--reverse', default=False, action='store_true', help="sort in reverse order")

def main(args):
    with open(args.json_file, 'r') as f:
        data = json.load(f)
    # cap: AR
    # recall: RMSNE
    if args.sort_by == 'AR':
        keys = [f'cap_0{i}' for i in range(1, 9)]
    elif args.sort_by == 'RMSNE':
        keys = [f'recall_0{i}' for i in range(1, 9)]

    # Define a function to calculate the average AUC
    def average_auc(performance):
        return sum(performance['performance'][key] for key in keys) / len(keys)

    # Sort the data
    sorted_data = sorted(data, key=average_auc, reverse=args.reverse)

    # Print the average AUC for each performance
    for performance in sorted_data:
        avg_auc = average_auc(performance)
        print(f'Average {args.sort_by}: {avg_auc}')

    highest_auc_performance = sorted_data[0]
    highest_auc = average_auc(highest_auc_performance)
    print(f'Highest Average {args.sort_by}: {highest_auc}')
    for key in keys:
        print(key, highest_auc_performance["performance"][key])
    print(f'Corresponding hyperparameters: {highest_auc_performance["h_params"]}')


    # Now sorted_data is sorted by the average AUC in descending order
    with open('valid_metrics_pred_best_weights.json', 'w') as f:
        json.dump(sorted_data, f, indent=4)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)