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
    --subsamples_separator=\> \
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
    print_champion_trees(smc)
    logging.info("Tree found:")
    logging.info(smc.optimal_tree.to_str(reverse=True))

    logging.info("Results saved in: %s" % args.folder)


def run_smc_lcb(X):
    from g4l.smc_lcb import SMC
    logging.info("Estimating champion trees:")
    prune = SMC(args.max_depth, cache_dir=args.folder)
    prune.fit(X)
    print_champion_trees(prune)

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
    logging.info("Tree found:")
    logging.info(tree_found.to_str(reverse=True))



if __name__ == '__main__':

    args = smc_argparser()
    set_log(args.log_file, args.log_level)

    sample_cache_file = os.path.join(args.folder, 'sample.pkl')
    sample = Sample(args.sample_path.name, None, args.max_depth,
                    perl_compatible=bool(args.perl_compatible),
                    cache_file=sample_cache_file,
                    subsamples_separator=args.split)
    methods_dict = {'bic': run_smc_bic, 'lcb': run_smc_lcb}
    methods_dict[args.method](sample)




