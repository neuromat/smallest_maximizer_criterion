#!/usr/bin/env python

"""
Runs estimators for the given parameters

Example:

[SMC]

python smc.py -d 4 \
    -s examples/linguistic_case_study/folha.txt \
    -f ../test/results/bp \
    -j 0 \
    -p 4 \
    --num_cores 4 \
    bic

"""

import os
from g4l.estimators.smc import SMC
from g4l.estimators.prune import Prune
from g4l.data import Sample
from g4l.util.command_line_methods import smc_argparser, set_log
from g4l.util.command_line_methods import get_num_cores, write_champion_trees
from g4l.util.command_line_methods import generate_report, n_sizes
import logging


def run_smc_bic(X):
    # Instantiates SMC object with received parameters
    smc = SMC(args.max_depth,
              penalty_interval=tuple(args.penalty_interval),
              epsilon=args.epsilon,
              cache_dir=args.folder,
              df_method=args.df,
              perl_compatible=bool(args.perl_compatible))
    perform_estimation(smc)


def run_smc_lcb(X):
    # Instantiates LCB object with received parameters
    lcb = Prune(args.max_depth, cache_dir=args.folder)
    perform_estimation(lcb, X, args)


def perform_estimation(meth, X):
    # Estimates champion trees
    meth.fit(X)

    # Save champion trees to files
    write_champion_trees(meth)

    # Select the optimal model amongst the champion trees using bootstrap
    find_optimal_tree(meth, args)

    # Generates report
    generate_report(meth, X, args)


def find_optimal_tree(meth, args):
    logging.info("Finding optimal tree:")
    tree_found, opt_idx = meth.optimal_tree(args.resamples,
                                            n_sizes(),
                                            args.alpha,
                                            args.renewal_point,
                                            num_cores=get_num_cores(args))
    # Saves optimal tree into file
    tree_found.save(os.path.join(args.folder, 'optimal.tree'))
    logging.info("Tree found:")
    logging.info(tree_found.to_str(reverse=True))
    logging.info("Results saved in: %s" % args.folder)


if __name__ == '__main__':
    args = smc_argparser()
    set_log(args.log_file, args.log_level)
    sample_cache_file = os.path.join(args.folder, 'sample.pkl')
    logging.info("Estimating champion trees...")
    {
        'bic': run_smc_bic,
        'lcb': run_smc_lcb
    }[args.method](Sample(args.sample_path.name,
                          None, args.max_depth,
                          subsamples_separator=args.split,
                          perl_compatible=bool(args.perl_compatible),
                          cache_file=sample_cache_file))



