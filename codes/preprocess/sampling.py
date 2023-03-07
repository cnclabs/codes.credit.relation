import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--file_path', default='../data/interim/merged.csv',
                    help="Path of merged.csv")
parser.add_argument('--output_dir', default='../data/interim/sample/',
                    help="Directory of output data")
parser.add_argument('--mode', default='random', help="sampling methods")
parser.add_argument('--sample_time', default=500, help="sample_time")

if __name__ == '__main__':
    args = parser.parse_args()

    file_path = args.file_path
    # mode = args.mode

    merged_df = pd.read_csv(file_path, header=0)

    sample_id = pd.DataFrame(merged_df.id.unique(), columns=['id'])
    sample_id = sample_id.id.sample(n=int(args.sample_time), random_state=1, replace = False)

    sample_merge_df = merged_df[merged_df['id'].isin(sample_id)]

    # Save file
    sample_merge_path = args.output_dir # save path (folder)
    sample_merge_df.to_csv(sample_merge_path + "merged.csv", index=False)
    sample_merge_df.id.to_csv(sample_merge_path + "all_company_id.csv", index=False)