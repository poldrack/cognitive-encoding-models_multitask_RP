# cognitive encoding model analysis
# RP version

import pickle
import pandas as pd 
from pathlib import Path
import pathlib
import json
from encoding_model import EncodingModel


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

    # temp for testing, will replace with sklearn leave n out splitter
    train_index = [i for i in range(sub_df.shape[0] - 2)]
    test_index = [i for i in range(sub_df.shape[0] - 2, sub_df.shape[0])]
    train_X, test_X = sub_ontology_df.iloc[train_index, :], sub_ontology_df.iloc[test_index, :]
    train_Y, test_Y = sub_df.iloc[train_index, :], sub_df.iloc[test_index, :]
    # confirm indices are aligned
    assert(all(train_X.index == train_Y.index))
    assert(all(test_X.index == test_Y.index))

    # use the sorted list of held out tasks as the index for the model
    held_out_tasks = tuple(test_X.index.sort_values())
    encoding_models[held_out_tasks] = EncodingModel(method)
    encoding_models[held_out_tasks].fit(train_X, train_Y)
    
    return(encoding_models)


if __name__ == "__main__":

    datadir_base = Path("/Users/poldrack/Dropbox/code/cognitive-encoding-models_multitask")
    mapdir = datadir_base / 'results/em-0/parcellated_maps'
    resourcedir = datadir_base / 'resources'

    ontology_df = load_ontology(resourcedir)

    maps = load_parcellated_maps(mapdir, 1)

    subcodes = get_subcodes(maps)

    encoding_models = {}

    for subcode in subcodes:
        sub_df = get_sub_df(maps, subcode)
        sub_ontology_df = get_aligned_ontology(sub_df, ontology_df)
        encoding_models[subcode] = fit_encoding_models_l2o(sub_df, sub_ontology_df)
