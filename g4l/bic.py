import logging

from .smc_base import EstimatorsBase
from .context_tree import ContextTree
from .util import degrees_of_freedom as dof
import numpy as np
from tqdm import tqdm


class BIC(EstimatorsBase):
    """
    Implements the BIC tree estimator described in:
    Context tree selection using the Smallest Maximizer Criterion
    Galves, A., Galves, C., García, J. E., Garcia, N. L., & Leonardi, F. (2010).
    Context tree selection and linguistic rhythm retrieval from written texts.
    Annals of Applied Statistics, 4(1), 186–209.
    https://doi.org/10.1214/11-AOAS511

    Download available at https://arxiv.org/abs/0902.3619

    Originally presented by Csiszár and Talata (2006)
    in the paper 'Context tree estimation for not necessarily finite memory
    processes, Via BIC and MDL'

    ...

    Attributes
    ----------
    context_tree : list
        Champion tree found by the estimator

    Methods
    -------
    fit(X)
        Estimates champion trees for the given sample X and constant c
    """

    def __init__(self, c, df_method='perl', keep_data=False, perl_compatible=False):
        """
        Parameters
        ----------
        c : tuple
            The penalization constant used in the BIC estimator
        df_method : str
            The method used to calculate degrees_of_freedom. Options:
            - 'perl': uses the same df as the original implementation in perl
            - 'g4l': uses the method as described in the paper (slightly different)
            - 'csizar_and_talata': uses df as described in Csizar and Talata (2006)
        scan_offset : int
            Computes node frequencies starting from `scan_offset` position
        perl_compatible : bool
            Set algorithm to be compatible with the paper's perl version
        """
        assert c >= 0, 'c must be at least zero'
        self.c = c
        self.df_method = df_method
        self.keep_data = keep_data
        self.perl_compatible = perl_compatible

    def fit(self, X):
        """ Estimates Context Tree model using BIC """
        full_tree = ContextTree.init_from_sample(X, force_admissible=False)
        df, transition_probs = full_tree.df, full_tree.transition_probs
        # likelihood_pen => n^{-c \dot df(w)}L_{w}(X^{n}_{1})

        deg_f = dof.degrees_of_freedom(self.df_method, full_tree)
        penalty = penalty_term(X.len(), self.c, deg_f)
        df['likelihood_pen'] = penalty + df.likelihood
        full_tree.df = assign_values(X.max_depth, df[df.freq >= 1], self.perl_compatible)
        full_tree.prune_unique_context_paths()
        self.context_tree = clean_columns(full_tree, self.keep_data)
        return self


def calc_sum_v_children(df, level, max_depth):
    parents = df[(df.depth == level) & (df.freq >= 1)]
    ch = df[(df.parent_idx.isin(parents.node_idx)) & (df.freq >= 1)]
    ch_vals = ch.groupby([df.parent_idx]).v_sum.sum()
    ch_vals.name = 'v_sum'
    ch_vals = ch_vals.to_frame().reset_index()
    ch_vals.rename(columns={'parent_idx': 'node_idx'}, inplace=True)
    ch_vals = ch_vals.set_index('node_idx')
    df = df.set_index('node_idx').combine_first(ch_vals)
    return df.reset_index(drop=False)


def assign_values(max_depth, df, comp=False):
    # v_node = V^{c}_{w}(X^{n}_{1})
    df = df.copy()
    df.loc[df.depth == max_depth, 'v_node'] = df.likelihood_pen
    df.loc[df.depth == max_depth, 'v_node_sum'] = df.likelihood_pen
    df['active'] = 0
    df['indicator'] = 0
    df.set_index('node_idx', inplace=True)

    for d in reversed(range(0, max_depth)):
        parents = df[(df.depth == d) & (df.freq >= 1)]
        ch = df[(df.parent_idx.isin(parents.index)) & (df.freq >= 1)]
        ch_vals = ch.groupby([df.parent_idx]).v_node.sum()
        df.loc[ch_vals.index, 'v_node_sum'] = ch_vals
        df.loc[ch_vals.index, 'v_node'] = df.loc[ch_vals.index][['likelihood_pen', 'v_node_sum']].max(axis=1)
        df['comp_aux'] = df.v_node_sum > df.v_node
    cond = (df.depth < max_depth) & (df.v_node > df.likelihood_pen)
    # indicator => \delta_{w}^{c}(X^{n}_{1})
    df.loc[cond, 'indicator'] = 1

    if comp == True:
        # Make the code compatible with the perl version
        # this one fits condition in line 581
        # ( $TREE[ $l + 1 ][ ( $m + $j ) ][2] > 1 )
        for i, r in df.iterrows():
            ch = df[df.parent_idx == i]
            df.loc[df.index == i, 'num_child_nodes'] = len(ch[ch.freq > 1])
        df.loc[(cond & (df.num_child_nodes <= 1)), 'indicator'] = 0
        df.loc[(df.depth == 1) & (df.indicator == 0), 'active'] = 1
    else:
        if df[df.node == ''].indicator.values[0] == 0:
            df.loc[df.node == '', 'active'] = 1
            return df

    nd = df.set_index('node')
    for d in range(max_depth + 1):
        candidate_nodes = df.loc[(df.depth == d) & (df.indicator == 0)]
        itr = candidate_nodes.iterrows()
        if logging.getLogger().level == logging.DEBUG:
            itr = tqdm(itr, total=len(candidate_nodes))
        for idx, row in itr:
            node_suffixes = [row.node[-(d - m):] for m in range(1, d)]
            if comp == False:
                node_suffixes += ['']
            suffixes = nd.loc[node_suffixes]
            if suffixes['indicator'].product() == 1 and row.indicator == 0:
                df.loc[idx, 'active'] = 1
    return df


def penalty_term(sample_len, c, degr_freedom):
    return np.log(sample_len) * c * degr_freedom


def clean_columns(t, keep_data):
    logging.debug('Cleaning...')
    """
    Removes non-relevant info
    """
    if not keep_data:
        t.df = t.tree()
    node_idxs = t.df.node_idx.unique()
    tr = t.transition_probs.set_index('idx').loc[node_idxs]
    tr = tr[tr.prob > 0]
    t.transition_probs = tr
    return t
