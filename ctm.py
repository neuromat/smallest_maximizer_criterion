#!/usr/bin/env python

"""
Runs CTM/BIC for the given parameters

Example:


ctm -s fixtures/sample20000.txt -c 0.5 -d 6  ./output.tree


"""


import argparse
import os
from pathlib import Path
from g4l.estimators.bic import BIC
from g4l.data import Sample
import logging


def dir_path_force(temp_folder):
    return dir_path(temp_folder, force=True)


def dir_path(temp_folder, force=False):
    if force:
        Path(temp_folder).mkdir(parents=True, exist_ok=True)
    if os.path.isdir(temp_folder):
        return temp_folder
    else:
        raise NotADirectoryError(temp_folder)


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


def run_bic(X):
    logging.info("Estimating BIC tree:")
    keep = bool(args.keep)
    if args.check_consistency == 1:
        keep = True
    bic = BIC(args.penalty, args.max_depth,
              scan_offset=args.scan_offset,
              df_method=args.df,
              keep_data=keep,
              perl_compatible=args.perl_compatible)
    bic.fit(X)
    tree = bic.context_tree
    logging.info("Tree found:")
    logging.info(tree.to_str(reverse=True))
    filename = os.path.abspath(args.output.name)
    tree.save(filename)
    logging.info("File saved at: %s" % filename)
    if args.check_consistency == 1:
        try:
            from g4l.models import integrity
            integrity.is_freq_consistent(tree)
        except AssertionError as e:
            logging.error("Consistency problems detected in the resulting tree")
            logging.error(str(e))

    if args.inspect == 1:
        import code; code.interact(local=dict(globals(), **locals()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Estimates context tree')
    #subparsers = parser.add_subparsers(dest='action', help='Action')
    #subparsers.required = True

    #parser_builder = subparsers.add_parser('fit', help='Create model given a sample')
    parser.add_argument('-c', '--penalty', type=float, help='Penalty constant')
    parser.add_argument('-d', '--max_depth',
                        type=int,
                        required=True,
                        help='Max tree depth')
    parser.add_argument('-s', '--sample_path',
                        type=argparse.FileType('r'),
                        required=True,
                        help='Sample path')
    parser.add_argument('-k', '--keep',
                        type=int,
                        default=0,
                        help='Set 1 if you want to keep the full nodes details')
    parser.add_argument('--split',
                        type=str,
                        default=None,
                        help='Split sample character')
    parser.add_argument('--check_consistency',
                        type=int,
                        default=0,
                        help='Check consistency')
    parser.add_argument('--inspect',
                        type=int,
                        default=0,
                        help='Inspect tree')
    parser.add_argument('--scan_offset',
                        type=int,
                        default='0',
                        help='Start reading sample from this index on')
    parser.add_argument('--perl_compatible',
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
    parser.add_argument('output', type=argparse.FileType('w'), help='output tree file (ex. my_model.tree)')
    args = parser.parse_args()
    set_log(args.log_file, args.log_level)
    run_bic(Sample(args.sample_path.name, None, args.max_depth,
                   perl_compatible=args.perl_compatible,
                   subsamples_separator=args.split))
