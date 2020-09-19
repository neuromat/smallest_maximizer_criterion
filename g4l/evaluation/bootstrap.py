import random
import pandas as pd
import numpy as np
import g4l

class Bootstrap():
  sample = None
  partition_string = None
  resamples = None

  def __init__(self, sample, partition_string=None):
    self.sample = sample
    self.partition_string = partition_string
    self.resamples = dict()

  def resample(self, num_resamples, size=None):
    return Resample(self.sample,
                    num_resamples,
                    size = size,
                    partition_string = self.partition_string)



class Resample():
  def __init__(self, source_sample, num_resamples, size=None, partition_string=None):
    self.source_sample = source_sample
    self.num_resamples = num_resamples
    self.partition_string = partition_string
    if partition_string==None:
      self.partition_string = self.__most_frequent_substring(source_sample)
    if size==None:
      self.size = len(source_sample.data)
    else:
      self.size = int(size)

  def generate(self):
    for resample_idx, resample in self.__iter(self.num_resamples):
      yield resample_idx, resample

  def __iter(self, num_resamples):
    data = self.source_sample.data
    slices = np.array(np.char.split([data], self.partition_string)[0])
    num_slices = len(slices)
    for idx, i in enumerate(range(num_resamples)):
      shuffled_indexes = np.random.randint(num_slices, size=num_slices*3)
      new_data = ''.join(slices[shuffled_indexes])
      yield idx, new_data[:self.size]


  def __most_frequent_substring(self, source_sample):
    # TODO: implementation
    return '0'
