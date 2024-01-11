import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import click


@click.command()
@click.option('--f_name', default=None, help='The file to calculate.')
@click.option('--o_name', default=None, help='The file to output.')
def main(f_name, o_name):
    df = pd.read_csv(f_name)
    label_col = [col for col in df.columns if 'y' in col]
    result_df = df.loc[:, label_col]
    result_df = result_df.replace({-1:0, 2:0})
    result_df = result_df.apply(pd.value_counts)
    result = result_df.T.reset_index()
    result['ratio'] = result[1] / result[0]
    result.to_csv(o_name, index=False)
    result['ratio'].plot()
    plt.savefig(o_name[:-4] + '.pdf')

if __name__ == "__main__":
    main()
