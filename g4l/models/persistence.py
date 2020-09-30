import h5py
import pandas as pd
from g4l.data import Sample

def load_model(filename):
  """ Loads model from file """
  with h5py.File(filename, 'r') as h5obj:
    sample = _read_sample(h5obj)
    max_depth = h5obj.attrs['max_depth']
  contexts = pd.read_hdf(filename, 'contexts')
  transition_probs = pd.read_hdf(filename, 'transition_probs')
  return sample, max_depth, contexts, transition_probs


def save_model(context_tree, filename):
  """ Saves the model into a file """
  context_tree.df.to_hdf(filename, 'contexts', mode='w')
  context_tree.transition_probs.to_hdf(filename, 'transition_probs', mode='a')
  with h5py.File(filename, 'a') as h5obj:
    h5obj.attrs['max_depth'] = context_tree.max_depth
    _write_sample(h5obj, context_tree.sample)


def _write_sample(h5obj, sample):
  if sample is not None:
    if sample.filename:
      h5obj.attrs['sample.filename'] = sample.filename
    if sample.A:
      h5obj.attrs['sample.A'] = sample.A
    if sample.separator is not None:
      h5obj.attrs['sample.separator'] = sample.separator


def _read_sample(h5obj):
  return Sample(  try_key(h5obj, 'sample.filename'),
                  try_key(h5obj, 'sample.A'),
                  try_key(h5obj, 'sample.separator'))


def try_key(h5obj, attr_key):
  try:
    return h5obj.attrs[attr_key]
  except KeyError:
    return ''
