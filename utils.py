"""
utiility functions
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


def get_prediction_accuracy(predicted_maps, true_maps):
    """
    take in two nvoxels X 2 arrays
    return leave-two-out accuracy (i.e. r(p1,t1) > r(p1, t2) & r(p2, t2) > r(p2, t1)) and predictions
    """

    assert predicted_maps.shape[0] == 2 and true_maps.shape[0] == 2
    assert predicted_maps.shape[1] == true_maps.shape[1]

    corrs = np.zeros((2, 2))
    for true_map in range(2):
        for predicted_map in range(2):
            corrs[true_map, predicted_map] = np.corrcoef(true_maps[true_map, :],
                                                         predicted_maps[predicted_map, :])[0, 1]
    predictions = [np.argmax(corrs[:, 0]), np.argmax(corrs[:, 1])]

    accuracy = 1 if predictions == [0, 1] else 0

    return(accuracy, predictions)


def get_df_r2score(df1, df2):
    """
    compute r2score for each column between data frames
    first align by index
    """
    assert all(df1.columns == df2.columns)

    df1_copy, df2_copy = df1.copy(), df2.copy()
    df1_copy, df2_copy = df1_copy.align(df2_copy)
    r2score = {k: r2_score(df1[k].values, df2[k].values) for k in df1_copy.columns}

    return(pd.Series(r2score))


def get_regionwise_accuracy(predicted, sub_df):
    """
    compute r/r2 within each region across all contrasts

    measure (str): either 'r2' (for sklearn r2_score) or 'r' (for pearson r)
    """

    pred_array = []
    pred_keys = []
    # first generate a data frame with predicted for each contrast
    for k, pred in predicted.items():
        pred_array.append(pred[0, :])
        pred_keys.append(k[0])
        pred_array.append(pred[1, :])
        pred_keys.append(k[1])
    pred_df = pd.DataFrame(pred_array, index=pred_keys)

    # get mean prediction for each contrast
    mean_prediction_per_contrast = {}
    for contrast in pred_df.index.unique():
        contrast_df = pred_df.query(f'index == "{contrast}"')
        mean_prediction_per_contrast[contrast] = contrast_df.mean(axis=0)

    mean_prediction_df = pd.DataFrame(mean_prediction_per_contrast).T
    return({'r': mean_prediction_df.corrwith(sub_df, axis=0),
            'r2': get_df_r2score(mean_prediction_df, sub_df)})
