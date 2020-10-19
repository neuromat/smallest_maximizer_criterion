from .builders import incremental
from . import persistence as per
from collections import Counter
from .builders.tree_builder import ContextTreeBuilder
import numpy as np
import math
import pandas as pd
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
        self.df.loc[self.df.num_child_nodes.isna(), 'active'] = 1

    @classmethod
    def init_from_sample(cls, X, max_depth, initialization_method=incremental,
                         force_admissible=True):
        """ Builds a full initial tree from a given sample """

        contexts, transition_probs = initialization_method.run(X, max_depth)
        t = ContextTree(max_depth, contexts, transition_probs, X)
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
        fn = lambda x: [m.end(0)+1 for m in re.finditer(x, sample.data, overlapped=True)]
        convert_to_symbols_fn = lambda x: [sample.data[i] for i in x if i < len(sample.data)]
        next_symbols_positions = contexts_in_resample.node.apply(fn)
        symbol_freqs = next_symbols_positions.apply(convert_to_symbols_fn)
        symbol_freqs = symbol_freqs[symbol_freqs.str.len()>0]
        contexts_in_resample['symbol_freqs'] = symbol_freqs.apply(lambda x: [Counter(x)[s] for s in sample.A])
        contexts_in_resample['count'] = symbol_freqs.apply(lambda x: len(x))
        builder = ContextTreeBuilder(sample.A)
        for idx, row in contexts_in_resample.iterrows():
            if hasattr(row.symbol_freqs, "__len__"):
                builder.add_context(row.node, row.symbol_freqs)
        t = builder.build()
        incremental.calculate_likelihood(t.df, t.transition_probs)
        return t.log_likelihood(), t

    def prune_unique_context_paths(self):
        df = self.df
        while True:
            df = self.calculate_num_child_nodes(df)
            leaves = df.loc[~df.index.isin(df.parent_idx)]
            single_leaves_parents = df.loc[leaves.parent_idx]
            single_leaves_parents = single_leaves_parents[single_leaves_parents.num_child_nodes == 1]
            nodes_to_remove = df[df.parent_idx.isin(single_leaves_parents.index)]
            if len(nodes_to_remove) == 0:
                break
        df.loc[df.node_idx.isin(nodes_to_remove), 'active'] = 0
        #df.drop(index=nodes_to_remove.index, inplace=True)

    def calculate_num_child_nodes(self, df):
        num_child_nodes = df[df.depth > 1].reset_index(drop=False).groupby(['parent_idx']).apply(lambda x: x.count().node_idx)
        try:
            df['num_child_nodes'] = num_child_nodes
        except ValueError:
            df['num_child_nodes'] = 0
        return df

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

    def find_suffix(self, sample):
        contexts = self.tree().set_index('node')['node_idx']
        for i in range(1, self.max_depth+1):
            substr = sample[-i:]
            try:
                return substr, contexts.loc[substr]
            except KeyError:
                pass

    def generate_sample(self, sample_size, A):
        """ Generates a sample from this model """

        df = self.tree().set_index(['node_idx'])
        sample = df[df.depth==self.max_depth].sample()
        s = sample.node.values[0]
        node_idx = sample.index[0]
        while len(s) < sample_size:
            s += self._next_symbol(node_idx, A)
            _, node_idx = self.find_suffix(s)
        return s

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
        r = df[~df.node_idx.isin(df[df.active==1].parent_idx)]
        if active_only==True:
            r = r[r.active==1]
        return r

    def leaves(self):
        return np.sort(list(self.tree()['node']))

    def equals_to(self, context_tree):

        """ Matches the current context tree to another one """
        return self.to_str()==context_tree.to_str()

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

    def _next_symbol(self, node_idx, A):
        transitions = self.transition_probs.reset_index()
        transitions = transitions[transitions.idx==node_idx] #.set_index(['next_symbol'])
        ps = transitions.prob.values
        elements = transitions.next_symbol.values #loc[A].index
        return np.random.choice(elements, 1, p=ps)[0]
