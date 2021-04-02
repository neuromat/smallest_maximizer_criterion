import os
import pickle
import shutil
import tarfile
from tempfile import TemporaryDirectory
import tempfile

from ..sample import Sample
from pathlib import Path


def load_model(filename):
    """ Loads model from file """

    with tarfile.open(filename, mode='r:') as tar:
        contexts = load_tar_file(tar, 'contexts.pkl')
        transition_probs = load_tar_file(tar, 'transitions.pkl')
        mtd = load_tar_file(tar, 'metadata.pkl')

    sample = Sample(None, mtd['A'],
                    mtd['max_depth'],
                    data='',
                    subsamples_separator=mtd['sample_subseparator'])
    sample.filename = mtd['sample_filename']
    return sample, mtd['max_depth'], contexts, transition_probs


def save_model(context_tree, filename):
    """ Saves the model into a file """

    with tempfile.TemporaryDirectory() as tmpdirname:
        context_tree.df.to_pickle(os.path.join(tmpdirname, 'contexts.pkl'))
        transitions_path = os.path.join(tmpdirname, 'transitions.pkl')
        context_tree.transition_probs.to_pickle(transitions_path)
        metadata = {
            'A': context_tree.sample.A,
            'max_depth': context_tree.max_depth,
            'sample_filename': context_tree.sample.filename,
            'sample_subseparator': context_tree.sample.subsamples_separator
        }
        with open(os.path.join(tmpdirname, 'metadata.pkl'), 'wb') as mfile:
            pickle.dump(metadata, mfile)
        make_tarfile(filename, tmpdirname)


def load_tar_file(tar, file):
    f = [x for x in tar.getmembers() if x.name == file][0]
    return pickle.load(tar.extractfile(f))


def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w") as tar:
        for filename in os.listdir(source_dir):
            tar.add(os.path.join(source_dir, filename), arcname=filename)


def _write_sample(h5obj, sample):
    if sample is not None:
        if sample.filename:
            h5obj.attrs['sample.filename'] = sample.filename
        if sample.A:
            h5obj.attrs['sample.A'] = sample.A


def create_temp_folder(temp_folder):
    Path(temp_folder).mkdir(parents=True, exist_ok=True)


def tempdir():
    temp_cache = TemporaryDirectory()
    return temp_cache.name


def save_champion_trees(context_trees, cache_dir):
    trees_folder = os.path.join(cache_dir, 'champion_trees')
    try:
        shutil.rmtree(trees_folder)
    except FileNotFoundError:
        pass
    create_temp_folder(trees_folder)
    for i, tree in enumerate(context_trees):
        n = '%06d' % i
        tree.save(os.path.join(trees_folder, '%s.tree' % n))
    return trees_folder

