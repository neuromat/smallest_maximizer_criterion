#!/usr/bin/env python

"""
Runs CTM/BIC for the given parameters

usage: ctm.py [-h] [-c PENALTY] -d MAX_DEPTH -s SAMPLE_PATH [-k KEEP] [--split SPLIT] [--check_consistency CHECK_CONSISTENCY] [--inspect INSPECT]
              [--perl_compatible PERL_COMPATIBLE] [--df {ct06,perl,g4l}] [--num_cores NUM_CORES] [-l LOG_FILE] [-i {quiet,debug,info,warning,error}]
              output

Estimates context tree

positional arguments:
  output                output tree file (ex. my_model.tree)

optional arguments:
  -h, --help            show this help message and exit
  -c PENALTY, --penalty PENALTY
                        Penalty constant
  -d MAX_DEPTH, --max_depth MAX_DEPTH
                        Max tree depth
  -s SAMPLE_PATH, --sample_path SAMPLE_PATH
                        Sample path
  -k KEEP, --keep KEEP  Set 1 if you want to keep the full nodes details
  --split SPLIT         Split sample character
  --check_consistency CHECK_CONSISTENCY
                        Check consistency
  --inspect INSPECT     Inspect tree
  --perl_compatible PERL_COMPATIBLE
                        Keeps compatibility with original version in perl (def. False)
  --df {ct06,perl,g4l}  Penalization strategy
  --num_cores NUM_CORES
                        Number of processors for parallel processing
  -l LOG_FILE, --log_file LOG_FILE
                        Log file path
  -i {quiet,debug,info,warning,error}, --log_level {quiet,debug,info,warning,error}
                        Log level


Example:

python ctm.py -s fixtures/sample20000.txt -c 0.5 -d 6  ../output.tree


"""


import logging
from g4l.bic import BIC
from g4l.sample import Sample
from g4l.util.command_line_methods import ctm_argparser, set_log
from g4l.util.command_line_methods import keep, save_file, check_consistency


def run_ctm(X, args):
    logging.info("Estimating BIC tree:")
    # Instantiates BIC object with received parameters
    bic = BIC(args.penalty,
              df_method=args.df,
              keep_data=keep(args),
              perl_compatible=args.perl_compatible)
    bic.fit(X)
    tree = bic.context_tree
    logging.info("Tree found:")
    logging.info(tree.to_str(reverse=True))
    save_file(tree, args)
    check_consistency(tree, args)



if __name__ == '__main__':
    args = ctm_argparser()
    set_log(args.log_file, args.log_level)
    sample = Sample(args.sample_path.name, None, args.max_depth,
                    perl_compatible=args.perl_compatible,
                    subsamples_separator=args.split)
    run_ctm(sample, args)
