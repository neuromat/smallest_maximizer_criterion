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


import argparse
import os
from pathlib import Path
from g4l.estimators.smc import SMC
from g4l.estimators.prune import Prune
from g4l.data import Sample
from g4l.reports.smc import SmcReport
import logging
import numpy as np


def dir_path_force(folder):
    return dir_path(folder, force=True)


def dir_path(folder, force=False):
    if force:
        Path(folder).mkdir(parents=True, exist_ok=True)
    if os.path.isdir(folder):
        return folder
    else:
        raise NotADirectoryError(folder)


def log_levels():
    return {
        'quiet': None,
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR
    }


def set_log(log_file=None, lvl='info'):
    if lvl == 'quiet':
        return
    log_handlers = []
    if log_file:
        log_handlers.append(logging.FileHandler(log_file))
    else:
        log_handlers.append(logging.StreamHandler())

    logging.basicConfig(
        level=log_levels()[lvl],
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=log_handlers
    )


def write_champion_trees(meth):
    logging.info("Champions tree found:")
    for i, tree in enumerate(meth.context_trees):
        try:
            used_c = meth.thresholds[i]
        except:
            used_c = '-'
        logging.info("c:%s\t%s" % (used_c, tree.to_str()))


def run_smc_bic(X):
    logging.info("Estimating champion trees:")
    smc = SMC(args.max_depth,
              penalty_interval=tuple(args.penalty_interval),
              epsilon=args.epsilon,
              cache_dir=args.folder,
              scan_offset=args.scan_offset,
              df_method=args.df,
              perl_compatible=bool(args.perl_compatible))
    smc.fit(X)
    write_champion_trees(smc)

    logging.info("------------------------")
    logging.info("Finding optimal tree:")
    num_cores = 1
    if args.num_cores > 1:
        num_cores = args.num_cores
    n_sizes = tuple([int(len(X.data) * x) for x in args.n_sizes])
    tree_found, opt_idx = smc.optimal_tree(args.resamples,
                                           n_sizes,
                                           args.alpha,
                                           args.renewal_point,
                                           num_cores=num_cores)
    tree_found.save(os.path.join(args.folder, 'optimal.tree'))

    np.save(os.path.join(args.folder, 'bic_c'), smc.thresholds)
    SmcReport(args.folder).create_summary(smc, X, n_sizes, args)
    logging.info("Tree found:")
    logging.info(tree_found.to_str(reverse=True))

    logging.info("Results saved in: %s" % args.folder)


def run_smc_lcb(X):
    logging.info("Estimating champion trees:")
    prune = Prune(args.max_depth,
                  cache_dir=args.folder)
    prune.fit(X)
    write_champion_trees(prune)

    logging.info("------------------------")
    logging.info("Finding optimal tree:")
    num_cores = 1
    if args.num_cores > 1:
        num_cores = args.num_cores
    #import code; code.interact(local=dict(globals(), **locals()))
    n_sizes = tuple([int(len(X.data) * x) for x in args.n_sizes])
    tree_found, opt_idx = prune.optimal_tree(args.resamples,
                                             n_sizes,
                                             args.alpha,
                                             args.renewal_point,
                                             num_cores=num_cores)
    #import code; code.interact(local=dict(globals(), **locals()))
    logging.info("Tree found:")
    logging.info(tree_found.to_str(reverse=True))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Estimates context tree')
    subparsers = parser.add_subparsers(dest='method', help='Estimation method')
    subparsers.required = True

    parser_prune = subparsers.add_parser('lcb', help='Prune by the Less Contributive Branch')
    parser_prune.add_argument('-S', '--n_sizes',
                              nargs=2,
                              type=float,
                              metavar=('j1', 'j2'),
                              default=(0.3, 0.9),
                              help='Bootstrap sample sizes factor for j = 1, 2'
                              )
    parser_prune.add_argument('-b', '--resamples',
                              type=int,
                              default='200',
                              help='Number of bootstrap samples used')
    parser_prune.add_argument('-a', '--alpha',
                              type=float,
                              default='0.01',
                              help='Alpha value for t-test')

    parser_bic = subparsers.add_parser('bic', help='Smallest Maximizer Criterion parameters')
    parser_bic.add_argument('-S', '--n_sizes',
                            nargs=2,
                            type=float,
                            metavar=('j1', 'j2'),
                            default=(0.3, 0.9),
                            help='Bootstrap sample sizes factor for j = 1, 2'
                            )

    parser_bic.add_argument('-c', '--penalty_interval',
                            nargs=2,
                            type=float,
                            metavar=('pen_min', 'pen_max'),
                            default=(0, 100),
                            help='Penalization constant intervals for BIC',
                            )
    parser_bic.add_argument('-e', '--epsilon',
                            type=float,
                            default='0.01',
                            help='SMC stop condition value')
    parser.add_argument('--split',
                        type=str,
                        default=None,
                        help='Split sample character')
    parser.add_argument('-b', '--resamples',
                        type=int,
                        default='200',
                        help='Number of bootstrap samples used')
    parser.add_argument('-a', '--alpha',
                        type=float,
                        default='0.01',
                        help='Alpha value for t-test')
    parser.add_argument('-d', '--max_depth',
                        type=int,
                        required=True,
                        help='Max tree depth')
    parser.add_argument('-p', '--renewal_point',
                              type=str,
                              default=None,
                              help='Renewal point')
    parser.add_argument('-s', '--sample_path',
                        type=argparse.FileType('r'),
                        required=True,
                        help='Sample path')
    parser.add_argument('-f', '--folder',
                        type=dir_path_force,
                        default='.',
                        help='Folder path for result files')
    parser.add_argument('-o', '--scan_offset',
                        type=int,
                        default='0',
                        help='Start reading sample from this index on')
    parser.add_argument('-j', '--perl_compatible',
                        type=bool,
                        default=False,
                        help='Keeps compatibility with original version in perl (def. False)')
    parser.add_argument('--df',
                        choices=['csizar_and_talata', 'perl', 'g4l'],
                        default='perl',
                        help='Penalization strategy')
    parser.add_argument('--num_cores',
                        type=int,
                        default=0,
                        help='Number of processors for parallel processing')
    parser.add_argument('-l', '--log_file',
                        type=argparse.FileType('w'),
                        default=None,
                        help='Log file path')
    parser.add_argument('-i', '--log_level',
                        type=str,
                        choices=list(log_levels().keys()),
                        default='info',
                        help='Log level')

    args = parser.parse_args()
    set_log(args.log_file, args.log_level)
    sample_cache_file = os.path.join(args.folder, 'sample.pkl')
    {
        'bic': run_smc_bic,
        'lcb': run_smc_lcb
    }[args.method](Sample(args.sample_path.name,
                          None, args.max_depth,
                          subsamples_separator=args.split,
                          perl_compatible=bool(args.perl_compatible),
                          cache_file=sample_cache_file))



