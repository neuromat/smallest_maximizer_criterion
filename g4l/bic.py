from .smc_base import EstimatorsBase
from .tree_tables import calculate_num_child_nodes
from .context_tree import ContextTree
from .util import degrees_of_freedom as dof
from .estimators import pl_compat as pl
import numpy as np
import logging
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
            - 'ct06': uses df as described in Csizar and Talata (2006)
            (check g4l.util.degrees_of_freedom for further details)
        keep_data : bool
            When set to True, tree files are generated with all internal nodes
            (for inspection/analysis purposes)
        perl_compatible : bool
            Set algorithm to be compatible with the paper's perl version
        """
        assert c >= 0, 'c must be at least zero'
        self.c = c
        self.df_method = df_method
        self.keep_data = keep_data
        self.perl_compatible = perl_compatible

    def fit(self, X):
        """Estimates Context Tree model using BIC

        Arguments:
            X {Sample} -- A Sample object

        Returns:
            self
        """

        tree = ContextTree.init_from_sample(X)
        self._penalize_likelihoods(X, tree)
        self._prune(tree, X.max_depth, self.perl_compatible)
        calculate_num_child_nodes(tree.df)
        self.context_tree = self._clean_columns(tree, self.keep_data)
        return self

    def _penalize_likelihoods(self, X, tree):
        """Creates a column 'likelihood_pen' containing
           the penalized log-likelihoods

        Arguments:
            X {Sample} -- A sample object
            tree {ContextTree} -- The initial context tree object
        """
        nodes_df = tree.df
        sample_len = X.len()
        deg_f = dof.degrees_of_freedom(self.df_method, tree)
        penalty = self._penalty_term(sample_len, self.c, deg_f)
        nodes_df['likelihood_pen'] = penalty + nodes_df.likelihood

    def _penalty_term(self, sample_len, c, degr_freedom):
        """Returns the penalty value to be applied to node likelihoods

        Arguments:
            sample_len {int} -- The sample length
            c {float} -- The penalization constant
            degr_freedom {float} -- Degrees of freedom

        Returns:
            float -- Penalization value
        """
        return np.log(sample_len) * c * degr_freedom

    def _prune(self, tree, max_depth, comp=False):
        """Prunes nodes based on the Bayesian Information Criteria (BIC)

        [description]

        Arguments:
            tree {ContextTree} -- The initial tree
            max_depth {int} -- The maximum node depth

        Keyword Arguments:
            comp {bool} -- Compatibility mode
                           (with original perl version) (default: {False})
        """

        # v_node = V^{c}_{w}(X^{n}_{1})

        self._add_pruning_columns(tree, max_depth)
        self._compare_likelihoods(tree, max_depth, comp)
        self._activate_nodes(tree, max_depth, comp)

    def _compare_likelihoods(self, tree, max_depth, comp):
        """Compares the sum of child likelihood values with the parent's value

        Arguments:
            tree {ContextTree} -- The initial tree
            max_depth {int} -- The maximum node depth
            comp {bool} -- Compatibility mode parameter for perl version
        """
        df = tree.df
        # iterates over node depths (from max_depth to 0)
        for d in reversed(range(0, max_depth)):
            parents = df[(df.depth == d) & (df.freq >= 1)]
            ch = df[(df.parent_idx.isin(parents.index)) & (df.freq >= 1)]
            ch_vals = ch.groupby([df.parent_idx]).v_node.sum()
            df.loc[ch_vals.index, 'v_node_sum'] = ch_vals

            # updates v_node with max(likelihood_pen, v_node_sum)
            l_compare = df.loc[ch_vals.index][['likelihood_pen', 'v_node_sum']]
            df.loc[ch_vals.index, 'v_node'] = l_compare.max(axis=1)

        # sets indicator to 1 when v_node > likelihood_pen
        df.loc[(df.v_node > df.likelihood_pen), 'indicator'] = 1

    def _activate_nodes(self, tree, max_depth, comp):
        """Determines which nodes are contexts

        Arguments:
            tree {ContextTree} -- The initial tree
            max_depth {int} -- The maximum node depth
            comp {bool} -- Compatibility mode parameter for perl version
        """

        df = tree.df
        if not comp:
            # activates the empty node if indicator is 0
            # thus, the selected tree will be the empty tree
            if df[df.node == ''].indicator.values[0] == 0:
                df.loc[df.node == '', 'active'] = 1
                tree.df = df
                return
        else:
            # makes the algorithm compatible with the perl version
            pl.activate_first_depth_nodes(df, max_depth)

        nd = df.set_index('node')
        for d in range(max_depth + 1):
            candidate_nodes = df.loc[(df.depth == d) & (df.indicator == 0)]
            for idx, row in self._nodes_iterator(candidate_nodes):
                node_suffixes = [row.node[-(d - m):] for m in range(1, d)]
                if comp is False:
                    node_suffixes += ['']
                suffixes = nd.loc[node_suffixes]
                if suffixes['indicator'].product() == 1 and row.indicator == 0:
                    df.loc[idx, 'active'] = 1
        tree.df = df

    def _add_pruning_columns(self, tree, max_depth):
        """Adds auxiliary columns for pruning criteria

            The columns are created as follows.
            - v_node      => v_node = V^{c}_{w}(X^{n}_{1}) (LaTeX)
            - v_node_sum  => sum of children v_nodes
            - indicator   => $\\delta_{w}^{c}(X^{n}_{1})$

            Example of the new fields created:

                        node  active       v_node   v_node_sum  indicator
            node_idx
            6         001010       0 -1053.421818 -1053.421818          0
            7          01010       0          NaN          NaN          0
            8           1010       0          NaN          NaN          0
            9            010       0          NaN          NaN          0
            10            10       0          NaN          NaN          0
            11             0       0          NaN          NaN          0

        Arguments:
            tree {ContextTree} -- The initial tree
            max_depth {int} -- Maximum node depth
        """

        df = tree.df[tree.df.freq >= 1]
        df.loc[df.depth == max_depth, 'v_node'] = df.likelihood_pen
        df.loc[df.depth == max_depth, 'v_node_sum'] = df.likelihood_pen
        df['active'] = 0
        df['indicator'] = 0
        df.set_index('node_idx', inplace=True)
        tree.df = df

    def _nodes_iterator(self, candidate_nodes):
        """
        Auxiliary method to produce a visual-friendly iterator for the nodes
        """

        itr = candidate_nodes.iterrows()
        if logging.getLogger().level == logging.DEBUG:
            itr = tqdm(itr, total=len(candidate_nodes))
        return itr

    def _clean_columns(self, tree, keep_data):
        """ Removes non-relevant info """

        # removes non-context nodes unless keep_data is True
        if not keep_data:
            tree.df = tree.tree()

        # removes rows from transitions table where
        # transition probability is 0
        node_idxs = tree.df.node_idx.unique()
        tr = tree.transition_probs.set_index('idx').loc[node_idxs]
        tr = tr[tr.prob > 0]
        tree.transition_probs = tr
        return tree
