"""
Methods to grant compatibility with the original perl version
"""

from collections import defaultdict
import numpy as np


def activate_first_depth_nodes(df, max_depth):
    # this method fits the condition in g4l.pl:581
    # the perl version doesn't consider the empty node
    # 1st depth nodes are set to active
    cond = (df.depth < max_depth) & (df.v_node > df.likelihood_pen)
    for i, r in df.iterrows():
        ch = df[df.parent_idx == i]
        df.loc[df.index == i, 'num_child_nodes'] = len(ch[ch.freq > 1])
    df.loc[(cond & (df.num_child_nodes <= 1)), 'indicator'] = 0
    df.loc[(df.depth == 1) & (df.indicator == 0), 'active'] = 1


def calculate_substr_frequencies(sample):
    from itertools import product
    import re
    from pandas import DataFrame

    def get_substring_count(s, sub_s):
        return sum(1 for m in re.finditer('(?=%s)' % sub_s, s))

    d = defaultdict(lambda: np.zeros(len(sample.A)))
    dfreq = defaultdict(lambda: 0)
    for r in range(sample.max_depth+1):
        nodes = [''.join(x) for x in list(product(sample.A, repeat=r))]
        for node in nodes:

            dfreq[node] = get_substring_count(sample.data, node)
            d[node] = [get_substring_count(sample.data, node + x) for x in sample.A]

    df = DataFrame.from_dict(d).T
    df['N'] = [dfreq[x] for x in list(df.index)]
    return df
