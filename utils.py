"""
utiility functions
"""

import numpy as np
import pandas as pd

def get_prediction_accuracy(predicted_maps, true_maps):
    """
    take in two nvoxels X 2 arrays
    return leave-two-out accuracy (i.e. r(p1,t1) > r(p1, t2) & r(p2, t2) > r(p2, t1))
    """

    assert predicted_maps.shape[0] == 2 and true_maps.shape[0] == 2
    assert predicted_maps.shape[1] == true_maps.shape[1]

    corrs = np.zeros((2, 2))
    for true_map in range(2):
        for predicted_map in range(2):
            corrs[true_map, predicted_map] = np.corrcoef(true_maps[true_map, :],
                                                            predicted_maps[predicted_map, :])[0, 1]
    if (corrs[0, 0] > corrs[1, 0]) and (corrs[1, 1] > corrs[0, 1]):
        accuracy = 1
    else:
        accuracy = 0
    
    return(accuracy)


def get_regionwise_r2score(predicted, sub_df):
    """
    compute r2 within each region across all contrasts
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
    # return columnwise (region) correlation between mean prediction and true data
    # corrwith() automatically aligns data frames before computing correlation
    return(mean_prediction_df.corrwith(sub_df, axis=0))
