import os
import sys
import click
from tqdm import trange
from subprocess import check_call

PYTHON = sys.executable

@click.command()
@click.option('--path', default='../data/processed/len_12')
@click.option('--mode', default='index')
def main(path, mode):
    for fold in trange(10):
        fold_string = "{}_fold_{:02d}".format(mode, fold + 1)
        file_name = 'train_cum.gz'
        input_path = os.path.join(path, fold_string)
        input_file = os.path.join(input_path, file_name)
        output_path = input_path
        cmd = "{python} data_splitter.py\
                --input_file {input_file}\
                --output_path {output_path}\
                --mode {mode}".format(
                        python=PYTHON,
                        input_file=input_file,
                        output_path=output_path,
                        mode=mode)
        print(cmd)
        check_call(cmd, shell=True)



if __name__ == "__main__":
    main()
