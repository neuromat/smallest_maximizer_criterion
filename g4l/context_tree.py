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
    def init_from_sample(cls, X):
        """Builds a full initial tree from a given sample

        Arguments:
            X {Sample} -- A sample object

        Returns:
            ContextTree -- An initial tree (with all internal nodes)
        """
        contexts, transition_probs = nodes_and_transitions(X)
        t = ContextTree(X.max_depth, contexts, transition_probs, X)
        contexts = calculate_num_child_nodes(contexts)
        contexts.loc[contexts.num_child_nodes.isna(), 'active'] = 1
        return t

    @classmethod
    def load_from_file(cls, file_path):
        """Loads model data from file

        Arguments:
            file_path {str} -- File path for an already estimated model

        Returns:
            ContextTree -- The loaded model
        """

        X, max_depth, contexts, transition_probs = load_model(file_path)
        return ContextTree(max_depth, contexts, transition_probs, X)

    def sample_likelihood(self, sample):
        """Returns the log-likelihood of the model to the given sample

        [description]

        Arguments:
            sample {Sample} -- A Sample object

        Returns:
            float -- A float value between -inf and 0
        """

        contexts = self.tree().node.values
        trn_freqs = sample.F[sample.F.index.isin(contexts)][[int(x) for x in sample.A]]
        node_freqs = trn_freqs.T.sum()
        N2 = trn_freqs.to_numpy()
        ind = N2 > 0
        pos_freqs = N2[ind]
        sum_freqs = repmat(node_freqs.values, len(sample.A), 1).T[ind]
        L = np.sum(np.multiply(pos_freqs, np.log(pos_freqs) - np.log(sum_freqs)))
        return L

    def to_str(self, reverse=False):
        """Represents context tree as a string

        [description]

        Keyword Arguments:
            reverse {bool} -- Reverses node orientation (default: {False})

        Returns:
            str -- ex. ''
        """

        ret = ' '.join([s.strip() for s in self.leaves()])
        if reverse:
            s1 = sorted([x[::-1] for x in ret.split()])
            s2 = [x[::-1] for x in s1]
            ret = ' '.join(s2)
        return ret.strip()

    def generate_sample(self, sample_size):

        """ Generates a sample using this model """
        A = self.sample.A
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
        return smpl[:sample_size]

    def num_contexts(self):
        """ Returns the number of contexts """

        return len(self.leaves())

    def log_likelihood(self):
        """ Returns the total log likelihood for all active contexts """

        return self.tree().likelihood.sum()

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

    def copy(self):
        """ Creates a complete copy of the model """

        return ContextTree(self.max_depth, self.df.copy(),
                           self.transition_probs.copy(),
                           source_sample=self.sample)

    def save(self, file_path):
        """ Saves model in a file

        Arguments:
            file_path {str} -- File path for an already estimated model
        """

        save_model(self, file_path)

    def _next_symbol(self, node_idx, A, trs):
        s = trs.loc[node_idx].sample(1, weights='prob').index[0]
        return A[s]
