from .tree_tables import nodes_and_transitions, calculate_num_child_nodes
from .util.persistence import load_model, save_model
import numpy as np
from numpy.matlib import repmat
from tqdm import tqdm


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

    @classmethod
    def init_from_sample(cls, X, force_admissible=True):
        """ Builds a full initial tree from a given sample """

        contexts, transition_probs = nodes_and_transitions(X)
        t = ContextTree(X.max_depth, contexts, transition_probs, X)
        contexts = calculate_num_child_nodes(contexts)
        contexts.loc[contexts.num_child_nodes.isna(), 'active'] = 1
        if force_admissible:
            t.prune_unique_context_paths()
        return t

    @classmethod
    def load_from_file(cls, file_path):
        """ Loads model data from file """

        X, max_depth, contexts, transition_probs = load_model(file_path)
        return ContextTree(max_depth, contexts, transition_probs, X)

    def copy(self):
        """ Creates a complete copy of the model """
        return ContextTree(self.max_depth, self.df.copy(),
                           self.transition_probs.copy(),
                           source_sample=self.sample)

    def save(self, file_path):
        """ Saves model in a file """

        save_model(self, file_path)

    #    def _count_context_freqs(self, node, sample_data, A):
    #        # get the index of each occurrence of each node
    #        idxs = [m.end(0) for m in re.finditer(node, sample_data, overlapped=True)]
    #        # count frequencies
    #        ctr = Counter([sample_data[i] for i in idxs if i < len(sample_data)])
    #        # return array with freqs for each symbol in alphabet
    #        return [ctr[a] for a in A]
    #
    #    def calculate_node_transitions(self, sample_data, A):
    #        all_contexts = list(self.df.node.values)
    #        trs = [self._count_context_freqs(ctx, sample_data, A) for ctx in all_contexts]
    #        transitions = np.hstack(trs).reshape(len(all_contexts), len(A))
    #        return all_contexts, transitions

    def sample_likelihood(self, sample):
        contexts = self.tree().node.values
        trn_freqs = sample.F[sample.F.index.isin(contexts)][[int(x) for x in sample.A]]
        node_freqs = trn_freqs.T.sum()
        N2 = trn_freqs.to_numpy()
        ind = N2 > 0
        pos_freqs = N2[ind]
        sum_freqs = repmat(node_freqs.values, len(sample.A), 1).T[ind]
        L = np.sum(np.multiply(pos_freqs, np.log(pos_freqs) - np.log(sum_freqs)))
        return L, (None, None)

    def prune_unique_context_paths(self):
        while True:
            df = calculate_num_child_nodes(self.df)
            leaves = df[(df.active_children == 0) & (df.active == 1)]
            parents_idx = [x for x in leaves.parent_idx.unique() if x not in [None, -1]]

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

    def __str__(self):
        return self.to_str()

    def to_str(self, reverse=False):
        """ Represents context tree as a string

        TODO: add `reverse=False` parameter to display contexts as root->leaf
        """
        ret = ' '.join([s.strip() for s in self.leaves()])
        if reverse:
            s1 = sorted([x[::-1] for x in ret.split()])
            s2 = [x[::-1] for x in s1]
            ret = ' '.join(s2)
        return ret.strip()

    def generate_sample(self, sample_size, A):
        """ Generates a sample using this model """
        trs = self.transition_probs.reset_index()
        trs.set_index(['idx', 'next_symbol'], inplace=True)
        contexts = self.tree().set_index('node')['node_idx']
        dd = self.tree().set_index(['node_idx'])
        if len(dd) == 0:
            return ''
        sample = dd[dd.depth == dd.depth.max()].sample()
        node_idx = sample.index[0]
        smpl = sample.node.values[0]
        for i in tqdm(range(sample_size)):
            symb = self._next_symbol(node_idx, A, trs)
            smpl += symb
            suffixes = [smpl[-i:] for i in range(1, self.max_depth+1)]
            node_idx = contexts[contexts.index.isin(suffixes)].iloc[0]
        return smpl

    def num_contexts(self):
        """ Returns the number of contexts """

        return len(self.leaves())

    def log_likelihood(self):
        """ Returns the total log likelihood for all active contexts """

        return self.tree().likelihood.sum()

    def node_transitions(self, contexts_only=False):
        """ Returns a table where each row is a node and each column is the
            transition probability for a symbol of the alphabet """

        import pandas as pd
        tbl = self.df
        if contexts_only:
            tbl = self.tree()
        nodes = tbl.set_index('node_idx').node
        dfx = self.transition_probs.merge(nodes,
                                          right_index=True,
                                          left_index=True)
        dfx = dfx.pivot(index='node',
                        columns='next_symbol',
                        values='prob').fillna(0).sort_values('node')
        return pd.DataFrame(dfx.to_records()).set_index('node')

    def tree(self):
        """ Returns the tree with all active contexts ascending by nodes"""

        return self.contexts().sort_values(by=['node'],
                                           ascending=(True))

    def contexts(self, active_only=True):
        """ Returns the tree with all active contexts"""
        return self.df[self.df.active == 1]

    def leaves(self):
        return np.sort(list(self.tree()['node']))

    def equals_to(self, context_tree):

        """ Matches the current context tree to another one """
        return self.to_str() == context_tree.to_str()

    def _next_symbol(self, node_idx, A, trs):
        s = trs.loc[node_idx].sample(1, weights='prob').index[0]
        return A[s]
