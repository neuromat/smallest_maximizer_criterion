#!/usr/bin/env python

from g4l.estimators.smc import SMC
from g4l.estimators.bic import BIC
import os
from g4l.data import Sample
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Create a sample object instance


#c = 160
#b = BIC(c, 4, scan_offset=0, df_method='perl', perl_compatible=True).fit(X).context_tree
#df = b.df
#df = df.drop('comp_aux', axis=1)
#print(df[df.depth <= 1] )
#print("BIC tree (c=%s):" % c, b.to_str())
#import code; code.interact(local=dict(globals(), **locals()))

max_depth = 4
num_cores = 6
num_resamples = 200
penalty_interval = (0.1, 400)
alpha = 0.01
renewal_point = '4'
perl_compatible = True
tmp_folder = '/home/arthur/tmp/smc1_ep'

X = Sample('examples/linguistic_case_study/publico.txt.bkp',
           [0, 1, 2, 3, 4],
           max_depth,
           cache_file=os.path.join(tmp_folder, 'ep.cache'),
           perl_compatible=perl_compatible,
           subsamples_separator='>')

#c = 0.34
#b = BIC(c, 4, df_method='perl', perl_compatible=perl_compatible, keep_data=True).fit(X).context_tree
#print("BIC tree (c=%s):" % c, b.to_str())
#import code; code.interact(local=dict(globals(), **locals()))
smc = SMC(max_depth,
          penalty_interval=penalty_interval,
          epsilon=0.01,
          cache_dir=tmp_folder,
          scan_offset=4,
          df_method='perl',
          perl_compatible=perl_compatible)
smc.fit(X)

t_hat, opt_idx = smc.optimal_tree(num_resamples,
                                  (int(X.len() * 0.3), int(X.len() * 0.9)),
                                  alpha,
                                  renewal_point,
                                  num_cores=num_cores)


for i, tree in enumerate(smc.context_trees):
    c = smc.thresholds[i]
    print("%s\t%s\t%s" % (tree.num_contexts(), c, tree.to_str()))

print("Optimal: " , t_hat.to_str(reverse=True) , " => ", t_hat.num_contexts())
# import code; code.interact(local=dict(globals(), **locals()))
#

#EP perl: '000 001 010 02 100 12 20 200 201 21 210 3 30 300 32 4 42 '
# '000 001 010 100 2 20 200 201 21 210 3 30 300 4'

# 0000 001 0010 100 2 20 200 2000 201 2010 21 210 3 30 300 4
# 0000 2000 100 200 300 0010 2010 210 20 30 001 201 21 2 3 4
