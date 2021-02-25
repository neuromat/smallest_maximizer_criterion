import sys, os
sys.path.insert(0, os.path.abspath('../..'))
from g4l.estimators.smc import SMC
from g4l.data import Sample
from g4l.bootstrap.resampling import BlockResampling
from g4l.bootstrap import Bootstrap
import numpy as np


def run_smc(X, cache_folder, instance_name='bp', num_cores=1, perl_compatible=True):
    max_depth = 4
    num_cores = 4
    num_resamples = 200
    penalty_interval = (0.1, 400)
    epsilon = 0.01
    alpha = 0.01
    renewal_point = '4'
    resamples_folder = '%s/resamples' % cache_folder
    L_path = "%s/L_%s.npy" % (resamples_folder, instance_name)
    resamples_file = "%s/resamples.%s.txt" % (resamples_folder, instance_name)
    data_len = X.len()
    n_sizes = (int(data_len * 0.3), int(data_len * 0.9))

    # Execute SMC to estimate champion trees
    smc = SMC(max_depth,
              penalty_interval=penalty_interval,
              epsilon=epsilon, scan_offset=0, df_method='perl',
              cache_dir=cache_folder, perl_compatible=perl_compatible)
    smc.fit(X)
    champion_trees = smc.context_trees

    t_hat, opt_idx = smc.optimal_tree(num_resamples,
                                      n_sizes,
                                      alpha,
                                      renewal_point,
                                      num_cores=num_cores)

    return champion_trees, opt_idx, smc
