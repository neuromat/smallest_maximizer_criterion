"""
usage: simulation_study.py [-h] [--model {model1,model2}] [--df {csizar_and_talata,perl,g4l}] [--resamples RESAMPLES] [--num_cores NUM_CORES]
                           [--samples_path SAMPLES_PATH] [--temp_folder TEMP_FOLDER] [--results_folder RESULTS_FOLDER] [--penalty_interval pen_min pen_max]
                           [--scan_offset SCAN_OFFSET] [--perl_compatible PERL_COMPATIBLE]
                           instance_name

Run simulation study

positional arguments:
  instance_name         Select model

optional arguments:
  -h, --help            show this help message and exit
  --model {model1,model2}
                        Select model
  --df {csizar_and_talata,perl,g4l}
                        penalization strategy
  --resamples RESAMPLES
                        number of bootstrap samples used
  --num_cores NUM_CORES
                        number of processors for parallel processing
  --samples_path SAMPLES_PATH
                        path containing the samples
  --temp_folder TEMP_FOLDER
                        path for temporary files
  --results_folder RESULTS_FOLDER
                        path for results
  --penalty_interval pen_min pen_max
                        Penalization constant intervals for BIC
  --scan_offset SCAN_OFFSET
                        start reading sample from this index on
  --perl_compatible PERL_COMPATIBLE
                        keeps compatibility with original version in perl
"""


import sys
import os
sys.path.insert(0, os.path.abspath('..'))
import math

import pandas as pd
from g4l.estimators import BIC
from g4l.estimators import SMC
from g4l.estimators import Prune
from g4l.data import persistence
from tqdm import tqdm
from g4l.bootstrap import Bootstrap
from g4l.bootstrap.resampling import BlockResampling
import logging
import argparse

#logging.basicConfig(level=logging.INFO)


parser = argparse.ArgumentParser(description='Run simulation study')
parser.add_argument('instance_name',
                    type=str,
                    help='Select model')
parser.add_argument('--model',
                    choices=['model1', 'model2'],
                    default='model1',
                    help='Select model')
parser.add_argument('--df',
                    choices=['csizar_and_talata', 'perl', 'g4l'],
                    default='csizar_and_talata',
                    help='penalization strategy')
parser.add_argument('--resamples',
                    type=int,
                    default='200',
                    help='number of bootstrap samples used')
parser.add_argument('--num_cores',
                    type=int,
                    default=0,
                    help='number of processors for parallel processing')
parser.add_argument('--samples_path',
                    type=str,
                    default='./simulation_study/samples',
                    help='path containing the samples')
parser.add_argument('--temp_folder',
                    type=str,
                    default='',
                    help='path for temporary files')
parser.add_argument('--results_folder',
                    type=str,
                    default='',
                    help='path for results')
parser.add_argument('--penalty_interval',
                    nargs=2,
                    type=float,
                    metavar=('pen_min', 'pen_max'),
                    default=(0, 100),
                    help='Penalization constant intervals for BIC',
                    )
parser.add_argument('--scan_offset',
                    type=int,
                    default='0',
                    help='start reading sample from this index on')
parser.add_argument('--perl_compatible',
                    type=bool,
                    default=False,
                    help='keeps compatibility with original version in perl')
args = parser.parse_args()


DF_METHOD = args.df
INSTANCE_NAME = args.instance_name
NUM_RESAMPLES = args.resamples
NUM_CORES = None if args.num_cores == 0 else args.num_cores
SCAN_OFFSET = args.scan_offset
PERL_COMPATIBLE = args.perl_compatible
samples_path = args.samples_path
ESTIMATORS = ['smc'] # ['smc', 'prune']

opt_tmp_fld = os.path.abspath('./simulation_study/instances/%s/tmp' % INSTANCE_NAME)
opt_results_folder = os.path.abspath('./simulation_study/instances/%s/results' % INSTANCE_NAME)
temp_folder = args.temp_folder if args.temp_folder != '' else opt_tmp_fld
results_folder = args.results_folder if args.results_folder != '' else opt_results_folder

A = ['0', '1']
SAMPLE_SIZES = [5000, 10000, 20000]
PENALTY_INTERVAL = tuple(args.penalty_interval)
RENEWAL_POINT = 1
N1_FACTOR = 0.3
N2_FACTOR = 0.9
C = 0.5
MAX_SAMPLES = 100  # math.inf
max_depth = 6

if not os.path.exists(results_folder):
    os.makedirs(results_folder, exist_ok=True)
    logging.info("Creating folder %s" % results_folder)
summary_file = '%s/summary.csv' % results_folder
all_vars = vars(args)
all_vars['c_min'] = args.penalty_interval[0]
all_vars['c_max'] = args.penalty_interval[1]
del(all_vars['penalty_interval'])
pd.DataFrame.from_dict(all_vars, orient='index').T.to_csv(summary_file, sep='\t', index=False)



def get_results_file(estimator, model_name, sample_size, results_folder):
    results_file = "%s/%s/%s_%s.csv" % (results_folder, estimator,
                                        model_name, sample_size)
    if os.path.exists(results_file):
        os.remove(results_file)
    return results_file


def run_simulation(model_name, temp_folder, results_folder, samples_path):
    estimators = {'prune': prune, 'smc': smc, 'bic': bic}
    logging.info("Running simulation with %s" % model_name)
    for sample_size in SAMPLE_SIZES:
        folder = "%s/%s/%s" % (temp_folder, model_name, sample_size)
        n_sizes = (sample_size * N1_FACTOR, sample_size * N2_FACTOR)
        print("Generating resamples:", folder)
        generate_bootstrap_resamples(model_name,
                                     sample_size,
                                     folder,
                                     n_sizes,
                                     samples_path)
        for estimator in ESTIMATORS:
            results_file = get_results_file(estimator, model_name, sample_size, results_folder)
            for sample_idx, sample in fetch_samples(model_name, sample_size, samples_path):
                print('sample:', sample_size, sample_idx)
                print("estimating champion trees")
                champion_trees = estimators[estimator](sample, temp_folder)
                resamples_file = resample_file(folder, sample_idx)
                bootstrap = Bootstrap(champion_trees, resamples_file, n_sizes)
                resamples_folder = '%s/likelihoods' % folder
                L = bootstrap.calculate_likelihoods(resamples_folder,
                                                    num_cores=NUM_CORES)
                opt_idx = bootstrap.find_optimal_tree(L, alpha=0.01)
                for tree_idx, champion_tree in enumerate(champion_trees):
                    opt = int(tree_idx == opt_idx)
                    obj = {'model_name': model_name,
                           'sample_idx': sample_idx,
                           'method': estimator,
                           'tree_idx': tree_idx,
                           'tree': champion_tree.to_str(),
                           'num_contexts': champion_tree.num_contexts(),
                           'likelihood': champion_tree.log_likelihood(),
                           'opt': opt,
                           'opt_idx': opt_idx}
                    use_header = (not os.path.exists(results_file))
                    df = pd.DataFrame.from_dict([obj])

                    outdir = os.path.dirname(results_file)
                    if not os.path.exists(outdir):
                        os.makedirs(outdir, exist_ok=True)

                    logging.info("Writing result to file %s" % results_file)
                    df.to_csv(results_file, mode='a',
                              index=False,
                              header=use_header)


def resample_file(folder, sample_idx):
    return "%s/resamples/%s.txt" % (folder, sample_idx)


def generate_bootstrap_resamples(model_name, sample_size, folder, larger_size, samples_path):
    args = (model_name, sample_size, samples_path, MAX_SAMPLES)
    for sample_idx, sample in tqdm(fetch_samples(*args), total=MAX_SAMPLES):
        file = resample_file(folder, sample_idx)
        resample_fctry = BlockResampling(sample, file, larger_size,
                                         RENEWAL_POINT)
        resample_fctry.generate(NUM_RESAMPLES, num_cores=NUM_CORES)


def resample_sizes(sample_size):
    return tuple(math.floor(f * sample_size) for f in [N1_FACTOR, N2_FACTOR])


def bic(sample, c, temp_folder):
    return [BIC(c, max_depth).fit(sample).context_tree]


def smc(sample, temp_folder):
    if temp_folder is None:
        cache_dir = None
    else:
        cache_dir = '%s/trees' % temp_folder

    smc = SMC(max_depth, penalty_interval=PENALTY_INTERVAL,
              epsilon=0.00001,
              df_method=DF_METHOD,
              scan_offset=SCAN_OFFSET,
              perl_compatible=PERL_COMPATIBLE,
              cache_dir=cache_dir)
    trees = smc.fit(sample).context_trees
    return sort_trees(trees)


def prune(sample, temp_folder):
    return sort_trees(Prune(max_depth).fit(sample).context_trees)


def sort_trees(context_trees):
    return sorted(context_trees, key=lambda x: -x.num_contexts())


def fetch_samples(model_name, sample_size, path, max_samples=math.inf):
    i = -1
    key = '%s_%s' % (model_name, sample_size)
    filename = '%s/%s.mat' % (path, key)
    for s in persistence.iterate_from_mat(filename, key, A):
        if i > max_samples:
            break
        i += 1
        yield i, s

run_simulation(args.model,
               temp_folder,
               results_folder,
               samples_path)
