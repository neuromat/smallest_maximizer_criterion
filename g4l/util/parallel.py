import math
import os
import shutil
import tqdm
import numpy as np
from multiprocessing import Pool
from g4l.models import ContextTree
from g4l.data import Sample
from itertools import product
from hashlib import md5


def calc_likelihood_process(args):
    trees_folder, resamples_file, resample_size, buf, tree_idx, resample_idx = args
    tree = get_tree(trees_folder, tree_idx)
    data = resamples(resamples_file)[resample_idx][:int(resample_size)]
    resample = Sample(None, tree.sample.A, data=data)

    ret, _ = tree.sample_likelihood(resample, buf=buf)
    return ret


def calculate_likelihoods(temp_folder, champion_trees, resamples_file,
                          resample_size, num_cores=None):
    num_resamples = len(resamples(resamples_file))
    num_trees = len(champion_trees)
    trees_folder = champion_trees_folder(temp_folder, champion_trees)
    persist_trees(champion_trees, trees_folder)
    pr = list(product(range(num_trees), range(num_resamples)))

    #import code; code.interact(local=dict(globals(), **locals()))
    samples = resamples(resamples_file)
    A = champion_trees[0].sample.A
    buf = [champion_trees[0].calculate_node_transitions(s[:resample_size], A) for s in samples]
    params = [(trees_folder, resamples_file, resample_size, buf[j], i, j) for i, j in pr]
    #import code; code.interact(local=dict(globals(), **locals()))
    if num_cores is None:
        result = map(calc_likelihood_process, params)
    else:
        with Pool(num_cores) as p:
            result = list(tqdm.tqdm(p.imap(calc_likelihood_process, params), total=len(params)))
    return np.reshape(list(result), (num_trees, num_resamples))


def get_tree(trees_folder, idx):
    return ContextTree.load_from_file('%s/tree_%s.h5' % (trees_folder, idx))


def resamples(resamples_file):
    with open(resamples_file) as f:
        ret = f.read().split('\n')[:-1]
    return ret


def champion_trees_folder(temp_folder, champion_trees):
    trees_str = ':'.join([s.to_str() for s in champion_trees])
    L = [str(math.floor(t.log_likelihood())) for t in champion_trees]
    trees_str += ''.join(L)
    trees_folder = md5((trees_str).encode('utf-8')).hexdigest()
    return '%s/trees_%s' % (temp_folder, trees_folder)


def remove_folder(trees_folder):
    if os.path.exists(trees_folder) and os.path.isdir(trees_folder):
        shutil.rmtree(trees_folder)


def persist_trees(champion_trees, trees_folder):
    remove_folder(trees_folder)
    os.makedirs(trees_folder)
    for idx, tree in enumerate(champion_trees):
        tree.save('%s/tree_%s.h5' % (trees_folder, idx))
