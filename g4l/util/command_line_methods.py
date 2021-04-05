import argparse
import os
from pathlib import Path
import logging


def ctm_argparser():  # pragma: no cover
    parser = argparse.ArgumentParser(description='Estimates context tree')
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
    parser.add_argument('--perl_compatible',
                        type=bool,
                        default=False,
                        help='Keeps compatibility with original version in perl (def. False)')
    parser.add_argument('--df',
                        choices=['ct06', 'perl', 'g4l'],
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
    return parser.parse_args()


def dir_path_force(temp_folder):
    return dir_path(temp_folder, force=True)


def dir_path(temp_folder, force=False):
    if force:
        Path(temp_folder).mkdir(parents=True, exist_ok=True)
    if os.path.isdir(temp_folder):
        return temp_folder
    else:
        raise NotADirectoryError(temp_folder)


def log_levels():  # pragma: no cover
    return {
        'quiet': None,
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR
    }


def set_log(log_file=None, lvl='info'):  # pragma: no cover
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


def get_num_cores(args):
    num_cores = 1
    if args.num_cores > 1:
        num_cores = args.num_cores
    return num_cores


def save_file(tree, args):
    filename = os.path.abspath(args.output.name)
    tree.save(filename)
    logging.info("File saved at: %s" % filename)


def check_consistency(tree, args):
    if args.check_consistency == 1:
        try:
            from g4l.models import integrity
            integrity.is_freq_consistent(tree)
        except AssertionError as e:
            msg = "Consistency problems detected in the resulting tree"
            logging.error(msg)
            logging.error(str(e))


def keep(args):
    keep = bool(args.keep)
    if args.check_consistency == 1:
        keep = True
    return keep


def smc_argparser():  # pragma: no cover
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
    parser.add_argument('-j', '--perl_compatible',
                        type=bool,
                        default=False,
                        help='Keeps compatibility with original version in perl (def. False)')
    parser.add_argument('--df',
                        choices=['ct06', 'perl', 'g4l'],
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
    return parser.parse_args()
