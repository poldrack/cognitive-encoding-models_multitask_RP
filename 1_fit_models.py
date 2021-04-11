# cognitive encoding model analysis
# RP version

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import pathlib
import json
from sklearn.model_selection import LeavePOut
from sklearn.metrics import r2_score
import sklearn
import argparse

from encoding_model import EncodingModel
from utils import get_prediction_accuracy, get_regionwise_r2score

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',"--verbose", help="increase output verbosity",
                        default=0, action='count')
    parser.add_argument('-m',"--method", help="regression method",
                            default='ridgecv')
    parser.add_argument('-o',"--ontology", help="ontology for model",
                            default='all')
    parser.add_argument('-s', "--shuffle", help="shuffle for null model",
                        action='store_true')
    return(parser.parse_args())


def load_parcellated_maps(datadir, sessioncode, nparcels=1000):
    assert sessioncode in (1, 2)
    if type(datadir) is not pathlib.PosixPath:
        datadir = Path(datadir)

    filename = f'all-subs_all-maps_{nparcels}-parcels_sc{sessioncode}.pkl'
    filepath = datadir / filename
    assert filepath.exists()

    with open(filepath, 'rb') as f:
        return(pickle.load(f))


def get_subcodes(maps):
    subcodes = list(maps.keys())
    subcodes.sort()
    return(subcodes)


def get_sub_df(maps, subcode):
    return(pd.DataFrame(maps[subcode]['zmaps'],
                        index=maps[subcode]['task_order']))


def load_ontology(resourcedir, ontology='all', remap=True,
                  remap_filepath=None):
    """
    remap: fix index of ontology to match subject data frame
    remap_filepath: path to json mapper file, defaults to task2task.json in resourcedir
    """

    ontology_file = resourcedir / f'X_{ontology}.csv'
    ontology_df = pd.read_csv(ontology_file, index_col=0)
    if remap:
        if remap_filepath is None:
            remap_filepath = resourcedir / 'task2task.json'
        with open(remap_filepath) as f:
            reverse_mapper = json.load(f)
            mapper = {reverse_mapper[i]: i for i in reverse_mapper}
            ontology_df.index = [mapper[i] for i in ontology_df.index]

    return(ontology_df)


def get_aligned_ontology(sub_df, ontology_df):
    # reorder ontology to match subject task ordering
    sub_ontology_df = ontology_df.reindex(sub_df.index)
    # confirm that indices match
    assert(all(sub_df.index == sub_ontology_df.index))
    return(sub_ontology_df)


def fit_encoding_models_l2o(sub_df, sub_ontology_df,
                            method='ridgecv', shuffle=False):
    """
    fit all encoding models using leave-2-out procedure

    Parameters:
        method: 'lr' (linear regression), 'ridgecv' (ridge regression)
        shuffle: if True, column-shuffle the ontology to create null model
    Returns:
        encoding_models (dict): a dict containing encoding model
        for each pair of held-out contrasts, for all ROIs
    """

    encoding_models = {}
    predicted_maps = {}

    # make a copy to prevent corruption of original if we shuffle
    sub_df_copy = sub_df.copy()

    splitter = LeavePOut(2)

    for train_index, test_index in splitter.split(sub_df):
        if shuffle:
            sub_df_copy = pd.DataFrame(sklearn.utils.shuffle(sub_df_copy.values), index=sub_df_copy.index)
        train_X, test_X = sub_ontology_df.iloc[train_index, :], sub_ontology_df.iloc[test_index, :]
        train_Y, test_Y = sub_df_copy.iloc[train_index, :], sub_df_copy.iloc[test_index, :]
        # confirm indices are aligned
        assert(all(train_X.index == train_Y.index))
        assert(all(test_X.index == test_Y.index))

        # use the list of held out tasks as the index for the model
        held_out_tasks = tuple(test_X.index)
        encoding_models[held_out_tasks] = EncodingModel(method)
        encoding_models[held_out_tasks].fit(train_X, train_Y)

        predicted_maps[held_out_tasks] = encoding_models[held_out_tasks].predict(test_X)

    return(encoding_models, predicted_maps)


def get_test_performance(predicted, sub_df):
    """
    compare predicted to actual maps across all pairs within subject
    """

    accuracy = {}
    r2score_mapwise = {}

    for key, predicted_maps in predicted.items():
        true_maps = sub_df.loc[key, :].values
        accuracy[key] = get_prediction_accuracy(predicted_maps, true_maps)

        r2score_mapwise[key] = [r2_score(true_maps[0, :], predicted_maps[0, :]),
                                r2_score(true_maps[1, :], predicted_maps[1, :])]

    # get regionwise r2score
    r2score_regionwise = get_regionwise_r2score(predicted, sub_df)

    return((accuracy, r2score_mapwise, r2score_regionwise))


def summarize_performance(performance):

    accuracy_by_subject = None
    r2_scores_mapwise_df = None
    r2_scores_regionwise = {}

    for subject, perf in performance.items():
        acc_values = np.array(list(perf[0].values()))
        acc_keys = list(perf[0].keys())

        if accuracy_by_subject is None:
            accuracy_by_subject = pd.DataFrame(
                acc_values.reshape((1, acc_values.shape[0])),
                columns=acc_keys, index=[subject])
        else:
            accuracy_by_subject = pd.concat(
                (accuracy_by_subject,
                 pd.DataFrame(acc_values.reshape((1, acc_values.shape[0])),
                              columns=acc_keys, index=[subject])))

        r2_values = np.array(list(perf[1].values()))
        r2_values = r2_values.reshape((r2_values.shape[0] * 2, 1))
        r2_keys_orig = list(perf[1].keys())
        #  need to duplicate these to correctly index the pairs of r2 scores
        pair_id = []
        map_id = []
        for k in r2_keys_orig:
            pair_id.append(k)
            pair_id.append(k)
            map_id.append(k[0])
            map_id.append(k[1])

        r2_subject = pd.Series(r2_values[:, 0]).to_frame('r2_score')
        r2_subject['subcode'] = subject
        r2_subject['pair_id'] = pair_id
        r2_subject['map_id'] = map_id

        if r2_scores_mapwise_df is None:
            r2_scores_mapwise_df = r2_subject
        else:
            r2_scores_mapwise_df = pd.concat((r2_scores_mapwise_df, r2_subject))

        r2_scores_regionwise[subject] = perf[2]
    accuracy_by_subject['subcode'] = accuracy_by_subject.index
    accuracy_df = pd.melt(accuracy_by_subject, id_vars='subcode')
    return(accuracy_df, r2_scores_mapwise_df, r2_scores_regionwise)


if __name__ == "__main__":

    datadir_base = Path("/Users/poldrack/Dropbox/code/cognitive-encoding-models_multitask")
    mapdir = datadir_base / 'results/em-0/parcellated_maps'
    resourcedir = datadir_base / 'resources'

    args = parse_args()

    method = args.method
    ontology = args.ontology
    shuffle = args.shuffle

    ontology_df = load_ontology(resourcedir, ontology)

    maps = load_parcellated_maps(mapdir, 1)

    subcodes = get_subcodes(maps)

    encoding_models = {}
    predicted_maps = {}
    performance = {}

    for subcode in subcodes:
        if shuffle:
            print('fitting null models for subject', subcode)
        else:
            print('fitting models for subject', subcode)
        sub_df = get_sub_df(maps, subcode)
        sub_ontology_df = get_aligned_ontology(sub_df, ontology_df)
        encoding_models[subcode], predicted_maps[subcode] = fit_encoding_models_l2o(
            sub_df, sub_ontology_df, method, shuffle)

        performance[subcode] = get_test_performance(predicted_maps[subcode], sub_df)

    if shuffle:
        ontology = 'null'

    outdir = Path('/Users/poldrack/data_unsynced/multitask/encoding_models')
    with open(outdir / f'encoding_models_{ontology}_{method}.pkl', 'wb') as f:
        pickle.dump(encoding_models, f)

    with open(outdir / f'predicted_maps_{ontology}_{method}.pkl', 'wb') as f:
        pickle.dump(predicted_maps, f)

    with open(outdir / f'peformance_{ontology}_{method}.pkl', 'wb') as f:
        pickle.dump(performance, f)

    accuracy_df, r2_scores_df, r2_scores_mapwise = summarize_performance(performance)
    accuracy_df.to_csv(outdir / f'accuracy_{ontology}_{method}.csv')
    r2_scores_df.to_csv(outdir / f'r2_mapwise_{ontology}_{method}.csv')
    with open(outdir / f'r2_regionwise_{ontology}_{method}.pkl', 'wb') as f:
        pickle.dump(r2_scores_mapwise, f)
    
