#!/usr/bin/env python

"""
Runs estimators for the given parameters


usage: smc.py [-h] [--split SPLIT] [-b RESAMPLES] [-a ALPHA] -d MAX_DEPTH [-p RENEWAL_POINT] -s SAMPLE_PATH [-f FOLDER] [-j PERL_COMPATIBLE]
              [--df {csizar_and_talata,perl,g4l}] [--num_cores NUM_CORES] [-l LOG_FILE] [-i {quiet,debug,info,warning,error}]
              {lcb,bic} ...

Estimates context tree

positional arguments:
  {lcb,bic}             Estimation method
    lcb                 Prune by the Less Contributive Branch
    bic                 Smallest Maximizer Criterion parameters

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
  -f FOLDER, --folder FOLDER
                        Folder path for result files
  -j PERL_COMPATIBLE, --perl_compatible PERL_COMPATIBLE
                        Keeps compatibility with original version in perl (def. False)
  --df {csizar_and_talata,perl,g4l}
                        Penalization strategy
  --num_cores NUM_CORES
                        Number of processors for parallel processing
  -l LOG_FILE, --log_file LOG_FILE
                        Log file path
  -i {quiet,debug,info,warning,error}, --log_level {quiet,debug,info,warning,error}
                        Log level

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
from g4l.sample import Sample
from g4l.bootstrap import Bootstrap
from g4l.util.command_line_methods import smc_argparser, set_log, get_num_cores
import logging


def run_smc(args, smc_method):
    X = Sample(args.sample_path.name,
               None, args.max_depth,
               subsamples_separator=args.split,
               perl_compatible=bool(args.perl_compatible),
               cache_file=os.path.join(args.folder, 'sample.pkl'))
    bootstrap = Bootstrap(args.folder, args.num_resamples, args.renewal_point)
    smc = smc_method.SMC(bootstrap,
                         penalty_interval=tuple(args.penalty_interval),
                         epsilon=args.epsilon,
                         cache_dir=args.folder,
                         df_method=args.df,
                         perl_compatible=bool(args.perl_compatible))
    smc.fit(X)
    smc.save_output(args.output)


def find_optimal_tree(smc, args):
    logging.info("Finding optimal tree:")
    num_cores = get_num_cores(args)
    tree_found, opt_idx = smc.optimal_tree(args.resamples,
                                           args.n_sizes,
                                           args.alpha,
                                           args.renewal_point,
                                           num_cores=num_cores)
    # Saves optimal tree into file
    tree_found.save(os.path.join(args.folder, 'optimal.tree'))
    logging.info("Tree found:")
    logging.info(tree_found.to_str(reverse=True))
    logging.info("Results saved in: %s" % args.folder)


def run_smc_bic(args):
    from g4l import smc_bic
    run_smc(args, smc_bic)


def run_smc_lcb(args):
    from g4l import smc_lcb
    run_smc(args, smc_lcb)


if __name__ == '__main__':
    args = smc_argparser()
    set_log(args.log_file, args.log_level)
    logging.info("Estimating champion trees...")
    {
        'bic': run_smc_bic,
        'lcb': run_smc_lcb
    }[args.method](args)



