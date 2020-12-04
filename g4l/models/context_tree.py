from .builders import incremental
from . import persistence as per
from collections import Counter
from .builders.tree_builder import ContextTreeBuilder
from .builders.resources import calculate_num_child_nodes
from numpy.matlib import repmat
import numpy as np
import math
import regex as re


class ContextTree():
    sample = None
    max_depth = None
    df = None
    transition_probs = None

    def __init__(self, max_depth, contexts_dataframe,
                 transition_probs, source_sample=None):
        self.max_depth = max_depth
        self.df = contexts_dataframe
        self.transition_probs = transition_probs
        self.sample = source_sample
        #self.df = calculate_num_child_nodes(self.df)
        #self.df.loc[self.df.num_child_nodes.isna(), 'active'] = 1

    @classmethod
    def init_from_sample(cls, X, max_depth, initialization_method=incremental,
                         force_admissible=True):
        """ Builds a full initial tree from a given sample """

        contexts, transition_probs = initialization_method.run(X, max_depth)
        t = ContextTree(max_depth, contexts, transition_probs, X)
        contexts = calculate_num_child_nodes(contexts)
        contexts.loc[contexts.num_child_nodes.isna(), 'active'] = 1
        if force_admissible:
            t.prune_unique_context_paths()
        return t

    @classmethod
    def load_from_file(cls, file_path):
        """ Loads model data from file """

        X, max_depth, contexts, transition_probs = per.load_model(file_path)
        return ContextTree(max_depth, contexts, transition_probs, X)

    def get_nodes(self, max_depth=math.inf):
        df = self.df
        df = df[df.depth <= max_depth]
        return df

    def copy(self):
        """ Creates a complete copy of the model """
        return ContextTree(self.max_depth, self.df.copy(),
                           self.transition_probs.copy(),
                           source_sample=self.sample)

    def save(self, file_path):
        """ Saves model in a file """

        per.save_model(self, file_path)

    def sample_likelihood(self, sample):
        contexts_in_resample = self.tree()[['node_idx', 'node']]
        fn = lambda x: [m.end(0) for m in re.finditer(x,
                                                      sample.data,
                                                      overlapped=True)]
        sample_len = len(sample.data)
        convert_fn = lambda x: [sample.data[i] for i in x if i < sample_len]
        next_symbols_positions = contexts_in_resample.node.apply(fn)
        transitions = next_symbols_positions.apply(convert_fn)
        N = np.zeros([len(contexts_in_resample), len(sample.A)])
        for i, tr in enumerate(transitions.values):
            counter = Counter(tr)
            for j, a in enumerate(sample.A):
                N[i, j] = counter[a]
        ss = np.array([sum(x) for x in N])
        ind = N > 0
        B = repmat(ss, len(sample.A), 1).T
        return np.sum(np.multiply(N[ind], np.log(N[ind]) - np.log(B[ind])))

    def prune_unique_context_paths(self):
        while True:
            df = calculate_num_child_nodes(self.df)
            leaves = df[(df.active_children == 0) & (df.active == 1)]
            parents_idx = [x for x in leaves.parent_idx.unique() if x is not None]
            #df[df.parent_idx]
            #leaves = df.loc[(~df.node_idx.isin(df.parent_idx)) & (df.active == 1)]
            lv_par = df.loc[parents_idx]  # single leaves' parents
            lv_par = lv_par[lv_par.num_child_nodes == 1]
            nodes_to_remove = df[df.parent_idx.isin(lv_par.index)]

            if len(nodes_to_remove) == 0:
                break
            l_nodes = self.df.node_idx.isin(nodes_to_remove.node_idx)
            l_parents = self.df.node_idx.isin(nodes_to_remove.parent_idx)
            self.df.loc[l_nodes, 'active'] = 0
            self.df.loc[l_parents, 'active'] = 1
            self.df = df
        #df.drop(index=nodes_to_remove.index, inplace=True)

    def evaluate_sample(self, new_sample):
        new_tree = self.copy()
        new_tree.df.node_freq = new_tree.df.likelihood = new_tree.df.ps = 0
        new_tree.df.transition_probs = None
        new_tree.df['transition_probs'] = self.df['transition_probs'].astype('object')
        new_tree.df.reset_index(drop=True, inplace=True)
        new_tree.calculate_node_frequency()
        new_tree.calculate_node_prob()
        return new_tree

    def __str__(self):
        return self.to_str()

    def to_str(self):
        """ Represents context tree as a string

        TODO: add `reverse=False` parameter to display contexts as root->leaf
        """

        return ' '.join(self.leaves())

    def generate_sample(self, sample_size, A):
        """ Generates a sample using this model """
        df = self.tree().set_index(['node_idx'])
        sample = df[df.depth == self.max_depth].sample()

        node_idx = sample.index[0]
        M = np.zeros((len(df), len(A)))
        trs = self.transition_probs.reset_index()
        for i in range(len(df)):
            for jj in range(len(A)):
                try:
                    M[i, jj] = trs[(trs.idx == i) & (trs.next_symbol == A[jj])].iloc[0].prob
                except IndexError:
                    M[i, jj] = 0
        contexts = self.tree().set_index('node')['node_idx']
        smpl = sample.node.values[0]
        while len(smpl) < sample_size:
            smpl += self._next_symbol(node_idx, A, M)
            suffixes = [smpl[-i:] for i in range(1, self.max_depth+1)]
            node_idx = contexts[contexts.index.isin(suffixes)].iloc[0]
        return smpl

    def num_contexts(self):
        """ Returns the number of contexts """

        return len(self.leaves())

    def log_likelihood(self):
        """ Returns the total log likelihood for all active contexts """

        return self.tree().likelihood.sum()

    def tree(self):
        """ Returns the tree with all active contexts ascending by nodes"""

        return self.contexts().sort_values(
                    by=['node'],
                    ascending=(True))

    def contexts(self, active_only=True):
        """ Returns the tree with all active contexts"""

        df = self.df
        r = df[~df.node_idx.isin(df[df.active == 1].parent_idx)]
        if active_only:
            r = r[r.active == 1]
        return r

    def leaves(self):
        return np.sort(list(self.tree()['node']))

    def equals_to(self, context_tree):

        """ Matches the current context tree to another one """
        return self.to_str() == context_tree.to_str()

    def calculate_node_frequency(self):
        for i, row in self.df.iterrows():
            self.df.at[i, 'freq'] = gen.calc_node_frequency(row.node, self.sample.data)

    def calculate_node_prob(self):
        sample_len = len(self.sample.data)
        transition_probs_idx = self.df.columns.get_loc('transition_probs')
        # calculates prob of occurrence for each node
        # self.df.ps = self.df.node_freq / (sample_len - self.df.l + 1)
        # calculates child nodes' probs
        for i, row in self.df.iterrows():
            fr, pb = gen.children_freq_prob(row.node, row.node_freq, self.A, self.sample.data)
            self.df.at[i, 'transition_probs'] = list(pb)
            self.df.at[i, 'likelihood'] = gen.calc_lpmls(fr, pb)

    def _next_symbol(self, node_idx, A, M):
        return np.random.choice(A, 1, p=M[node_idx])[0]
