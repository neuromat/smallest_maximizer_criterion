import pandas as pd
import numpy as np
import logging
from abc import ABCMeta, abstractmethod

class ResamplingBase():
  __metaclass__ = ABCMeta

  @abstractmethod
  def generate(self, sample, num_resamples, size, file):
    logging.error("generate method must be implemented!")

