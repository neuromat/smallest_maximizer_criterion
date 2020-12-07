import sys
import os
sys.path.insert(0, os.path.abspath('.'))
from g4l.models.builders.tree_builder import ContextTreeBuilder
from g4l.data import Sample
import pandas as pd
import numpy as np
import time
import logging

def get_model(model_name='model1'):
  """
  Generates models for the simulation described in the article [A. Galves
  et. al., Ann. Appl. Stat., Volume 6, Number 1 (2012), 186-209]

  Models:
    model1: The transition probabilities in model 1 were chosen with the purpose
    to make it difficult to find the true model with a small sample.

    model 2: The transition probabilities in model 2 were chosen to make it easy
    to find the model even with a relatively small sample.
  """
  model1_probs = [1.0, 0.3, 0.25, 0.2]
  model2_probs = [1.0, 0.2, 0.4, 0.3]
  models = {'model1': model1_probs, 'model2': model2_probs}
  return generate_model(models[model_name])


def generate_model(transition_probs):
  contexts = ['1', '10', '000', '100']
  A = ['0', '1']
  ctx_builder = ContextTreeBuilder(A)
  for i, context in enumerate(contexts):
    p = transition_probs[i]
    ctx_builder.add_context(context, np.array([p, 1-p])*1000)
  return ctx_builder.build()


