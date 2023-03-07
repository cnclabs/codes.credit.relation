import click
import numpy as np
import pandas as pd


np.seterr(divide='ignore', invalid='ignore')


def auc(x, y, reorder=False):
    """TODO: Docstring for auc.
    :returns: TODO
    """
    direction = 1
    if reorder:
        order = np.lexsort((y, x))
        x, y = x[order], y[order]
    else:
        dx = np.diff(x)
        if np.any(dx < 0):
            if np.all(dx <=0):
                direction = -1
            else:
                raise ValueError("x array is not increasing")

    area = direction * np.trapz(y, x)

    return area


def cap_curve(y_true, y_score):
    """TODO: Docstring for CAP_curve.
    Parameters
    -----
    y_true: array, shape = [n_samples]
    y_score: array, shape = [n_samples]
    :returns: TODO
    """
    pos_label = 1
    # y_true: boolean vector
    y_true = (y_true == pos_label)
    # sort scores and corresponding truth values
    desc_score_indicies = np.argsort(y_score, kind="mergesort")[::-1]

    y_score = y_score[desc_score_indicies]
    y_true = y_true[desc_score_indicies]


    # accumulate true-positive
    tps = np.cumsum(y_true)
    # accumulate total
    totals = np.cumsum(np.ones(y_true.shape))

    tpr = tps / tps[-1]
    totalr = totals / totals[-1]

    return tpr, totalr


def cap_ar_score(y_true, y_score):
    """TODO: Docstring for cap_ar_score.
    :returns: TODO
    """
    tpr_m, totalr_m = cap_curve(y_true, y_score)
    tpr_p, totalr_p = cap_curve(y_true, y_true)
    auc_m = auc(totalr_m, tpr_m)
    auc_p = auc(totalr_p, tpr_p)
    auc_r = auc(totalr_p, totalr_p)
    ar = (auc_m - auc_r)/(auc_p - auc_r)

    return ar


@click.command()
@click.option('--data_path', default=None, help='path to file folder')
@click.option('--month', default=None, help='path to file folder')
def main(data_path, month):
    """TODO: Docstring for main.
    :data_path: TODO
    :returns: TODO
    """
    df = pd.read_csv(data_path)
    #month = data_path.split('/')[-1].split('_')[0].strip('M')

    prob = df[month + 'M Default Prob']
    label = df[month + 'Month After Event']
    label.replace(2, 0, inplace=True)
    # select data, only consider 0, 1
    mask = np.logical_and((label.values != -1), (~np.isnan(prob.values)))
    y_score = prob[mask].values
    y_true = label[mask].values

    # testing data
    # y_score = np.linspace(0.05, 0.95, 10)
    # y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    ar = cap_ar_score(y_true, y_score)
    #print('AR: {0:.3f}'.format(ar))
    print("Month: "+month+"  Accuracy ratio: {0:.5f}".format(ar))


if __name__ == "__main__":
    main()