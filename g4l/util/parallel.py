import os
import shutil
import tqdm
import numpy as np
from multiprocessing import Pool
from g4l.context_tree import ContextTree
from g4l.sample import Sample
from itertools import product


def calc_likelihood_process(args):

    (trees_folder, resamples_file, resample_size,
     tree_idx, resample_idx, subsamples_separator) = args

    tree = get_tree(trees_folder, tree_idx)
    data = resamples(resamples_file)[resample_idx][:int(resample_size)]
    cache_file = os.path.join(buffer_folder(resamples_file),
                              str(resample_size),
                              '%s.cache' % resample_idx)
    resample = Sample(None, tree.sample.A,
                      tree.max_depth,
                      data=data,
                      subsamples_separator=tree.sample.subsamples_separator,
                      cache_file=cache_file)
    return tree.sample_likelihood(resample)


def buffer_folder(resamples_file):
    return os.path.join(os.path.dirname(resamples_file), 'resamples')


def calculate_likelihoods(champion_trees_folder,
                          resamples_file, resample_size,
                          subsamples_separator=None, num_cores=0):

    listfiles = list_files(champion_trees_folder)
    champion_trees = [ContextTree.load_from_file(f) for f in listfiles]
    num_resamples = len(resamples(resamples_file))
    num_trees = len(champion_trees)
    pr = list(product(range(num_trees), range(num_resamples)))

    params = [(champion_trees_folder,
               resamples_file,
               resample_size,
               i, j,
               subsamples_separator) for i, j in pr]

    if num_cores is None:
        num_cores = 0
    if num_cores > 1:
        with Pool(num_cores) as p:
            result = list(tqdm.tqdm(p.imap(calc_likelihood_process, params),
                          total=len(params)))
    else:
        result = list(tqdm.tqdm(map(calc_likelihood_process, params),
                      total=len(params)))

    return np.reshape(result, (num_trees, num_resamples))


def get_tree(trees_folder, idx):
    return ContextTree.load_from_file(list_files(trees_folder)[idx])


def list_files(trees_folder):
    tree_dir = os.listdir(trees_folder)
    return sorted([os.path.join(trees_folder, x) for x in tree_dir])


def resamples(resamples_file):
    return open(resamples_file).read().split('\n')[:-1]


def remove_folder(trees_folder):
    if os.path.exists(trees_folder) and os.path.isdir(trees_folder):
        shutil.rmtree(trees_folder)
