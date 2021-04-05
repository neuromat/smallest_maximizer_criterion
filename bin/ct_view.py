#!/usr/bin/env python

"""
Visualization tool for context trees

Example:

ct_view ./context_tree.tree


"""


import argparse
import os
from pathlib import Path
from g4l.models.context_tree import ContextTree
from g4l.display import plot2
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

# def plot2(tree, column_label='symbol', font_size=10,
#           label_offset_x=-14,
#           label_offset_y=10,
#           horizontal_alignment='right',
#           node_size=8,
#           linewidths=1.0,
#           node_color='black'):
#

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Visualizes context trees')
    parser.add_argument('--column',
                        type=str,
                        default='symbol',
                        help='Column to be shown in labels')
    parser.add_argument('--font_size',
                        type=int,
                        default=10,
                        help='Font size')
    parser.add_argument('tree_file', type=argparse.FileType('r'), help='Tree file')
    args = parser.parse_args()
    ff = args.tree_file.name
    tree = ContextTree.load_from_file(ff)
    plot2(tree, column_label=args.column, font_size=args.font_size)
