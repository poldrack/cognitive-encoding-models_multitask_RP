# cognitive encoding model analysis
# RP version

import pickle
import numpy as np
import pandas as pd 
from pathlib import Path
import pathlib
import json
from encoding_model import EncodingModel
from sklearn.model_selection import LeavePOut
from sklearn.metrics import r2_score


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


def fit_encoding_models_l2o(sub_df, sub_ontology_df, method='ridgecv'):
    """
    fit all encoding models using leave-2-out procedure

    Parameters:
        method: 'lr' (linear regression), 'ridgecv' (ridge regression)

    Returns:
        encoding_models (dict): a dict containing encoding model
        for each pair of held-out contrasts, for all ROIs
    """

    encoding_models = {}
    predicted_maps = {}

    splitter = LeavePOut(2)

    # temp for testing, will replace with sklearn leave n out splitter
    for train_index, test_index in splitter.split(sub_df):
        train_X, test_X = sub_ontology_df.iloc[train_index, :], sub_ontology_df.iloc[test_index, :]
        train_Y, test_Y = sub_df.iloc[train_index, :], sub_df.iloc[test_index, :]
        # confirm indices are aligned
        assert(all(train_X.index == train_Y.index))
        assert(all(test_X.index == test_Y.index))

        # use the sorted list of held out tasks as the index for the model
        held_out_tasks = tuple(test_X.index.sort_values())
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
        corrs = np.zeros((2, 2))
        for true_map in range(2):
            for predicted_map in range(2):
                corrs[true_map, predicted_map] = np.corrcoef(true_maps[true_map, :], 
                                                             predicted_maps[predicted_map, :])[0, 1]
        if (corrs[0, 0] > corrs[1, 0]) and (corrs[1, 1] > corrs[0, 1]):
            accuracy[key] = 1
        else:
            accuracy[key] = 0
        
        r2score_mapwise[key] = [r2_score(true_maps[0, :], predicted_maps[0, :]),
                                r2_score(true_maps[1, :], predicted_maps[1, :])]
    return((accuracy, r2score_mapwise))


if __name__ == "__main__":

    datadir_base = Path("/Users/poldrack/Dropbox/code/cognitive-encoding-models_multitask")
    mapdir = datadir_base / 'results/em-0/parcellated_maps'
    resourcedir = datadir_base / 'resources'

    method = 'ridgecv'
    ontology = 'all'

    ontology_df = load_ontology(resourcedir, ontology)

    maps = load_parcellated_maps(mapdir, 1)

    subcodes = get_subcodes(maps)

    encoding_models = {}
    predicted_maps = {}
    performance = {}

    for subcode in subcodes:
        print('fitting models for subject', subcode)
        sub_df = get_sub_df(maps, subcode)
        sub_ontology_df = get_aligned_ontology(sub_df, ontology_df)
        encoding_models[subcode], predicted_maps[subcode] = fit_encoding_models_l2o(
            sub_df, sub_ontology_df, method)

        performance[subcode] = get_test_performance(predicted_maps[subcode], sub_df)

    outdir = Path('/Users/poldrack/data_unsynced/multitask/encoding_models')
    with open(outdir / f'encoding_models_{method}.pkl', 'wb') as f:
        pickle.dump(encoding_models, f)

    with open(outdir / f'predicted_maps_{method}.pkl', 'wb') as f:
        pickle.dump(predicted_maps, f)
    
    with open(outdir / f'peformance_{method}.pkl', 'wb') as f:
        pickle.dump(performance, f)