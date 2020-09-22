import sys
import os
sys.path.insert(0, os.path.abspath('.'))
from g4l.models as tree
from g4l.data import Sample
import pandas as pd
import time
import logging

def model_one():
  generate_model([1.0, 0.3, 0.25, 0.2])

def model_two():
  generate_model([1.0, 0.2, 0.4, 0.3])

def generate_model():
  contexts = ['1', '01', '000', '001']
  A = ['0', '1']


