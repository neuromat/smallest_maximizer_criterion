import pandas as pd
from g4l.data import Sample
import os
import tempfile
import pickle
import tarfile


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


def load_tar_file(tar, file):
    f = [x for x in tar.getmembers() if x.name == file][0]
    return pickle.load(tar.extractfile(f))

def save_model(context_tree, filename):
    """ Saves the model into a file """

    with tempfile.TemporaryDirectory() as tmpdirname:
        context_tree.df.to_pickle(os.path.join(tmpdirname, 'contexts.pkl'))
        context_tree.transition_probs.to_pickle(os.path.join(tmpdirname, 'transitions.pkl'))
        metadata = {
            'A': context_tree.sample.A,
            'max_depth': context_tree.max_depth,
            'sample_filename': context_tree.sample.filename,
            'sample_subseparator': context_tree.sample.subsamples_separator
        }
        with open(os.path.join(tmpdirname, 'metadata.pkl'), 'wb') as mfile:
            pickle.dump(metadata, mfile)
        _make_tarfile(filename, tmpdirname)


def _make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w") as tar:
        for filename in os.listdir(source_dir):
            #import code; code.interact(local=dict(globals(), **locals()))
            tar.add(os.path.join(source_dir, filename), arcname=filename)


def _write_sample(h5obj, sample):
    if sample is not None:
        if sample.filename:
            h5obj.attrs['sample.filename'] = sample.filename
        if sample.A:
            h5obj.attrs['sample.A'] = sample.A
