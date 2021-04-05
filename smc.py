#!/usr/bin/env python

"""Runs estimators for the given parameters

usage: smc.py [-h] [--split SPLIT] [-b RESAMPLES] [-a ALPHA] -d MAX_DEPTH [-p RENEWAL_POINT] -s SAMPLE_PATH [-A ALPHABET] [-f FOLDER] [-j PERL_COMPATIBLE]
              [--df {ct06,perl,g4l}] [--num_cores NUM_CORES] [-l LOG_FILE] [-i {quiet,debug,info,warning,error}]
              {bic,lcb} ...

positional arguments:
  {bic,lcb}             Estimation method
    bic                 Prune using the Bayesian Information Criterion
    lcb                 Prune by the Less Contributive Branch

optional arguments:
  -h, --help            show this help message and exit
  --split SPLIT         Split sample character
  -b RESAMPLES, --resamples RESAMPLES
                        Number of bootstrap samples used
  -a ALPHA, --alpha ALPHA
                        Alpha value for t-test
  -d MAX_DEPTH, --max_depth MAX_DEPTH
                        Max tree depth
  -p RENEWAL_POINT, --renewal_point RENEWAL_POINT
                        Renewal point
  -s SAMPLE_PATH, --sample_path SAMPLE_PATH
                        Sample path
  -A ALPHABET, --alphabet ALPHABET
                        Symbols of the alphabet. Ex. '0 1 2 3 4'
  -f FOLDER, --folder FOLDER
                        Folder path for result files
  -j PERL_COMPATIBLE, --perl_compatible PERL_COMPATIBLE
                        Keeps compatibility with original version in perl (def. False)
  --df {ct06,perl,g4l}  Penalization strategy
  --num_cores NUM_CORES
                        Number of processors for parallel processing
  -l LOG_FILE, --log_file LOG_FILE
                        Log file path
  -i {quiet,debug,info,warning,error}, --log_level {quiet,debug,info,warning,error}
                        Log level


Example:

python smc.py -d 4 \
    -s examples/linguistic_case_study/folha.txt \
    -f ../test/results/bp \
    -A '0 1 2 3 4' \
    -p 4 \
    --split \> \
    --num_cores 4 \
    bic

"""


import os
import logging
from g4l.bootstrap import Bootstrap
from g4l.sample import Sample
from g4l.util.command_line_methods import smc_argparser, set_log


def print_champion_trees(meth):
    logging.info("Champions tree found:")
    for i, tree in enumerate(meth.context_trees):
        try:
            used_c = meth.thresholds[i]
        except:
            used_c = '-'
        logging.info("c:%s\t%s" % (used_c, tree.to_str()))


def run_smc_bic(X):
    from g4l.smc_bic import SMC
    num_cores = 1
    if args.num_cores > 1:
        num_cores = args.num_cores

    bootstrap = Bootstrap(args.folder, args.resamples, args.renewal_point)
    smc = SMC(bootstrap,
              penalty_interval=tuple(args.penalty_interval),
              cache_dir=args.folder,
              n_sizes=args.n_sizes,
              alpha=args.alpha,
              epsilon=args.epsilon,
              df_method=args.df,
              perl_compatible=bool(args.perl_compatible),
              num_cores=num_cores)
    smc.fit(X)
    report(smc, X)


def run_smc_lcb(X):
    from g4l.smc_lcb import SMC
    num_cores = 1
    if args.num_cores > 1:
        num_cores = args.num_cores

    bootstrap = Bootstrap(args.folder, args.resamples, args.renewal_point)
    smc = SMC(bootstrap,
              cache_dir=args.folder,
              n_sizes=args.n_sizes,
              alpha=args.alpha,
              num_cores=num_cores)
    smc.fit(X)
    report(smc, X)


def report(smc, X):
    print_champion_trees(smc)
    smc.save_output(X, args.folder)

    logging.info("Tree found:")
    logging.info(smc.optimal_tree.to_str(reverse=True))

    logging.info("Results saved in: %s" % args.folder)





if __name__ == '__main__':

    args = smc_argparser()
    set_log(args.log_file, args.log_level)

    sample_cache_file = os.path.join(args.folder, 'sample.pkl')
    A = args.alphabet
    if A:
        A = A.split(' ')
    sample = Sample(args.sample_path.name, A, args.max_depth,
                    perl_compatible=bool(args.perl_compatible),
                    cache_file=sample_cache_file,
                    subsamples_separator=args.split)
    methods_dict = {'bic': run_smc_bic, 'lcb': run_smc_lcb}
    methods_dict[args.method](sample)




