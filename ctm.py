#!/usr/bin/env python

"""
Runs CTM/BIC for the given parameters

Example:


python ctm.py -s fixtures/sample20000.txt -c 0.5 -d 6  ./output.tree

"""

import logging
import os

from g4l.bic import BIC
from g4l.sample import Sample
from g4l.util.command_line_methods import ctm_argparser, set_log


def run_ctm(X, args):
    logging.info("Estimating BIC tree:")
    # Instantiates BIC object with received parameters
    bic = BIC(args.penalty,
              df_method=args.df,
              keep_data=_keep(),
              perl_compatible=args.perl_compatible)
    bic.fit(X)
    tree = bic.context_tree
    logging.info("Tree found:")
    logging.info(tree.to_str(reverse=True))
    _save_file(tree)
    _check_consistency(tree)


def _save_file(tree):
    filename = os.path.abspath(args.output.name)
    tree.save(filename)
    logging.info("File saved at: %s" % filename)


def _check_consistency(tree):
    if args.check_consistency == 1:
        try:
            from g4l.models import integrity
            integrity.is_freq_consistent(tree)
        except AssertionError as e:
            msg = "Consistency problems detected in the resulting tree"
            logging.error(msg)
            logging.error(str(e))


def _keep():
    keep = bool(args.keep)
    if args.check_consistency == 1:
        keep = True
    return keep



if __name__ == '__main__':
    args = ctm_argparser()
    set_log(args.log_file, args.log_level)
    sample = Sample(args.sample_path.name, None, args.max_depth,
                    perl_compatible=args.perl_compatible,
                    subsamples_separator=args.split)
    run_ctm(sample, args)
