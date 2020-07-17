import numpy as np
import pandas as pd
from g4l.estimators.base import Base
from g4l.estimators.ctm import CTM
from datetime import datetime
import logging

class Prune():

  def execute(self, context_tree):
    estimator = PruneEstimator(context_tree)
    return estimator.execute()


class PruneEstimator(Base):

  def execute(self):
    import code; code.interact(local=dict(globals(), **locals()))

    # remove nodes where frequency is zero
    t = self.context_tree
    t.df.loc[t.df.ps==0, 'flag'] = 1
