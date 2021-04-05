import sys
import os
sys.path.insert(0, os.path.abspath('..'))
import math
import pandas as pd
from g4l.smc_bic import SMC
from g4l.util.mat import iterate_from_mat
from g4l.util import persistence as per
from g4l.bootstrap import Bootstrap
import tempfile

NUM_CORES = 4
CORRECT_TREE = '000 1 10 100'


class SimulationReport:
    def __init__(self, results_folder):
        self.folder = results_folder
        cols = ['model_name', 'sample_size', 'ct', 'sample_idx', 'opt']
        self.df = pd.DataFrame(columns=cols)
        per.create_temp_folder(self.folder)

    def add(self, smc, mdl, sample_size, sample_idx):
        opt = int(smc.optimal_tree.to_str() == CORRECT_TREE)
        arr = [mdl, sample_size, 1, sample_idx, opt]
        self.df.loc[len(self.df)] = arr
        self.persist()

    def summary(self):
        return self.df.groupby(['model_name', 'sample_size']).sum()[['ct', 'opt']]

    def persist(self):
        self.df.to_csv(os.path.join(self.folder, 'simulation.csv'),
                       index=False)


def fetch_samples(model_name, sample_size, path,
                  max_samples=math.inf, cache_dir=None):
    i = -1
    key = '%s_%s' % (model_name, sample_size)
    fl = '%s/%s.mat' % (path, key)
    for s in iterate_from_mat(fl, key, [0, 1], 6, cache_dir=cache_dir):
        if i > max_samples:
            break
        i += 1
        yield i, s


def run_simulation(tmp_path, results_path):
    report = SimulationReport(results_path)
    samples_path = './simulation_study/samples'
    for model_name in ['model1', 'model2']:
        for sample_size in [5000, 10000, 20000]:
            tmp = os.path.join(tmp_path, str(sample_size), str(model_name))
            for sample_idx, sample in fetch_samples(model_name,
                                                    sample_size,
                                                    samples_path,
                                                    cache_dir=tmp):
                bootstrap = Bootstrap(tmp, 100, '1')
                smc = SMC(bootstrap,
                          penalty_interval=(0.01, 800),
                          cache_dir=None,
                          n_sizes=(0.3, 0.9),
                          alpha=0.01,
                          epsilon=0.01,
                          df_method='ct06',
                          perl_compatible=False,
                          num_cores=NUM_CORES)
                smc.fit(sample)
                report.add(smc, model_name, sample_size, sample_idx)
                print(report.summary())
    print("============== Simulation results =======================")
    print(report.summary())

with tempfile.TemporaryDirectory() as tmp_path:
    run_simulation(tmp_path, './simulation_study/results')
