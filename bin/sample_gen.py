#!/usr/bin/env python

"""
Generates samples for a tree

Example:



python ctm.py -s tests/files/lipsum.txt -c 0 -d 6  ../lorem_ipsum.tree

python sample_gen.py -t ../lorem_ipsum.tree  -s 5000 ../lipsum_sample.txt

cat ../lipsum_sample.txt

"""


import argparse
import os
from pathlib import Path
from g4l.context_tree import ContextTree
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


def create_samples(args):
    tree = ContextTree.load_from_file(args.tree.name)
    x = tree.generate_sample(args.size) + '\n'
    fl = os.path.abspath(args.output_file.name)
    open(fl, 'w').write(x)
    logging.info("Sample '%s' generated" % fl)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Generates samples given a context tree')
    parser.add_argument('-t', '--tree',
                        type=argparse.FileType('r'),
                        required=True,
                        help='Context tree')
    parser.add_argument('-s', '--size',
                        type=int,
                        required=True,
                        help='Sample size')
    parser.add_argument('--num_cores',
                        type=int,
                        default=1,
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
    parser.add_argument('output_file', type=argparse.FileType('w'), default='./sample.out.txt', help='Output sample file.')

    args = parser.parse_args()
    set_log(args.log_file, args.log_level)
    create_samples(args)



