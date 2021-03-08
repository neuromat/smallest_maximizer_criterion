from .builders import incremental
from . import persistence as per
from collections import Counter
from .builders.resources import calculate_num_child_nodes
from numpy.matlib import repmat
import numpy as np
import math
import regex as re
from tqdm import tqdm
import logging


class ContextTree():
    sample = None
    max_depth = None
    df = None
    transition_probs = None

    def __init__(self, max_depth, contexts_dataframe,
                 transition_probs, source_sample=None, scan_offset=0):
        self.max_depth = max_depth
        self.df = contexts_dataframe
        self.transition_probs = transition_probs
        self.sample = source_sample
        self.scan_offset = scan_offset
        #self.df = calculate_num_child_nodes(self.df)
        #self.df.loc[self.df.num_child_nodes.isna(), 'active'] = 1

    @classmethod
    def init_from_sample(cls, X, max_depth, initialization_method=incremental,
                         force_admissible=True, scan_offset=0):
        """ Builds a full initial tree from a given sample """

        logging.debug("Initializing tree...")
        contexts, transition_probs = initialization_method.run(X, max_depth, scan_offset)
        t = ContextTree(max_depth, contexts, transition_probs, X, scan_offset)
        logging.debug("Calculating child nodes")
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

    def _count_context_freqs(self, node, sample_data, A):
        # get the index of each occurrence of each node
        idxs = [m.end(0) for m in re.finditer(node, sample_data, overlapped=True)]
        # count frequencies
        ctr = Counter([sample_data[i] for i in idxs if i < len(sample_data)])
        # return array with freqs for each symbol in alphabet
        return [ctr[a] for a in A]

    def calculate_node_transitions(self, sample_data, A):
        all_contexts = list(self.df.node.values)
        trs = [self._count_context_freqs(ctx, sample_data, A) for ctx in all_contexts]
        transitions = np.hstack(trs).reshape(len(all_contexts), len(A))
        return all_contexts, transitions

    def buffered_sample_likelihood(self, sample):
        #all_contexts, N = buf

        #contexts = list(self.tree().node.values)
        contexts = self.tree().node.values
        trn_freqs = sample.F[sample.F.index.isin(contexts)][[int(x) for x in sample.A]]
        node_freqs = trn_freqs.T.sum()

        #if N is None:
        #import code; code.interact(local=dict(globals(), **locals()))
        #freqs_trs = sample.freqs_and_transitions(self.max_depth)

        #all_nodes, N = self.calculate_node_transitions(sample.data, sample.A)
        #N2 = N[[all_nodes.index(ctx) for ctx in contexts]]
        #ss = np.array([sum(x) for x in N2])
        N2 = trn_freqs.to_numpy()
        ind = N2 > 0
        pos_freqs = N2[ind]

        sum_freqs = repmat(node_freqs.values, len(sample.A), 1).T[ind]
        L = np.sum(np.multiply(pos_freqs, np.log(pos_freqs) - np.log(sum_freqs)))

        return L, (None, None)

    def sample_likelihood(self, sample):
        #all_contexts, N = buf
        #contexts = list(self.tree().node.values)
        contexts = self.tree().node.values
        #if N is None:
        all_nodes, N = self.calculate_node_transitions(sample.data, sample.A)
        N2 = N[[all_nodes.index(ctx) for ctx in contexts]]
        ss = np.array([sum(x) for x in N2])
        ind = N2 > 0
        pos_freqs = N2[ind]

        sum_freqs = repmat(ss, len(sample.A), 1).T[ind]
        L = np.sum(np.multiply(pos_freqs, np.log(pos_freqs) - np.log(sum_freqs)))

        return L, (all_nodes, N)

    def prune_unique_context_paths(self):
        while True:
            df = calculate_num_child_nodes(self.df)
            leaves = df[(df.active_children == 0) & (df.active == 1)]
            parents_idx = [x for x in leaves.parent_idx.unique() if x not in [None, -1]]
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

    def to_ete(self):
        from ete3 import TreeNode

        def connect_node(node, dic, df, parent_idx):
            if parent_idx < 0:
                return
            parent_node = df.loc[parent_idx]
            try:
                parent = dic[parent_node.node]
            except KeyError:
                parent = TreeNode(name=parent_node.node)
                parent.add_feature('freq', parent_node.freq)
                parent.add_feature('context', parent_node.node)
                parent.add_feature('idx', parent_idx)
                dic[parent_node.node] = parent
            if len(parent.search_nodes(name=node.name)) == 0:
                parent.add_child(node)
            connect_node(parent, dic, df, parent_node.parent_idx)
        dic = dict()

        #root_node = self.df.set_index('node').loc['']
        root = TreeNode(name='')
        #root.add_feature('freq', root_node.freq)
        #root.add_feature('context', '')
        #root.add_feature('idx', root_node.node_idx)
        dic[''] = root

        df = self.df.set_index('node_idx')
        for i, x in self.tree().iterrows():
            l = TreeNode(name=x.node)
            l.add_feature('freq', x.freq)
            l.add_feature('context', x.node)
            l.add_feature('idx', x.node_idx)
            dic[x.node] = l
            connect_node(l, dic, df, x.parent_idx)
        return root

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

    def tree(self):
        """ Returns the tree with all active contexts ascending by nodes"""

        return self.contexts().sort_values(by=['node'],
                                           ascending=(True))

    def contexts(self, active_only=True):
        """ Returns the tree with all active contexts"""
        return self.df[self.df.active == 1]
        #df = self.df
        #r = df[~df.node_idx.isin(df[df.active == 1].parent_idx)]
        #if active_only:
        #    r = r[r.active == 1]
        #return r

    def leaves(self):
        return np.sort(list(self.tree()['node']))

    def equals_to(self, context_tree):

        """ Matches the current context tree to another one """
        return self.to_str() == context_tree.to_str()

    def calculate_node_frequency(self):
        for i, row in self.df.iterrows():
            self.df.at[i, 'freq'] = gen.calc_node_frequency(row.node,
                                                            self.sample.data,
                                                            self.scan_offset)

    def calculate_node_prob(self):
        for i, row in self.df.iterrows():
            fr, pb = gen.children_freq_prob(row.node, row.node_freq, self.A, self.sample.data)
            self.df.at[i, 'transition_probs'] = list(pb)
            self.df.at[i, 'likelihood'] = gen.calc_lpmls(fr, pb)

    def _next_symbol(self, node_idx, A, trs):
        #p = trs.loc[node_idx].prob
        #import code; code.interact(local=dict(globals(), **locals()))
        s = trs.loc[node_idx].sample(1, weights='prob').index[0]
        return A[s]
