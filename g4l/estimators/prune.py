import math
from g4l.models import ContextTree
from .base import CollectionBase
import logging

# Less contributive branches


class Prune(CollectionBase):

    def __init__(self, max_depth, max_trees=None, cache_dir=None):
        self.max_depth = max_depth
        self.max_trees = max_trees
        cache_dir = cache_dir or self._tempdir()
        super().__init__(cache_dir)
        assert(max_depth > 0)

    def fit(self, X):
        self.X = X
        t = ContextTree.init_from_sample(X, self.max_depth)
        self.trees_constructed = 0
        self.initialize_pruning(t)
        self.perform_pruning(t)
        #self.context_trees = list(reversed(self.context_trees))
        return self

    def initialize_pruning(self, t):
        df = t.df.copy()
        #df['children_contrib'] = df.transition_sum_log_probs
        df.loc[df.likelihood == 0, 'children_contrib'] = -math.inf
        df.loc[df.active == 0, 'num_total_leaves'] = 0
        df.loc[df.active == 0, 'num_direct_leaves'] = 0
        t.df = df

    def update_parent_counts(self, df, nodes_idx):
        updated_nodes = []
        nodes_to_update = df.loc[nodes_idx].sort_values(['depth'],
                                                        ascending=False)
        if len(nodes_to_update) == 0:
            return
        for depth in reversed(range(1, nodes_to_update.depth.max()+1)):
            cnd = (df.node_idx.isin(nodes_idx)) & (df.depth == depth)
            num_total_leaves = df.loc[cnd].groupby(['parent_idx'])
            num_total_leaves = num_total_leaves.sum().num_total_leaves
            idxs = list(num_total_leaves.index.values)
            current_leaves_count = df.loc[idxs].num_total_leaves.values
            df.loc[idxs, 'num_total_leaves'] = current_leaves_count + num_total_leaves.values
            updated_nodes += idxs
        self.update_parent_counts(df, updated_nodes)

    def update_counts(self, df):
        leaves = df[df.active == 1]
        leaf_counts = leaves.groupby(['parent_idx']).count().node_idx
        contrib = leaves.groupby(['parent_idx']).sum().likelihood
        df.loc[leaf_counts.index, 'num_total_leaves'] = leaf_counts
        df.loc[leaf_counts.index, 'num_direct_leaves'] = leaf_counts
        df.loc[leaf_counts.index, 'children_contrib'] = contrib
        self.update_parent_counts(df, leaf_counts.index.values)

    def perform_pruning(self, t):
        self.context_trees = []
        max_trees = self.max_trees
        iteration_num = 0
        self.add_tree(t)
        df = t.df.copy()
        while True:
            self.update_counts(df)
            cnd = ((df.num_total_leaves == df.num_direct_leaves) &
                   (df.num_total_leaves > 0) &
                   (df.depth > 0))
            candidate_nodes = df[cnd]
            if len(candidate_nodes) == 0:
                break
            child_cnd = ((df.parent_idx.isin(candidate_nodes.node_idx)) &
                         (df.active == 1))
            candidate_children = df[child_cnd]
            lps2 = candidate_children.groupby(['parent_idx']).likelihood.sum()
            diff = (lps2 - candidate_nodes.likelihood)
            less_contributive_node_idx = diff.sort_values().index[0]
            self.remove_leaves(df, less_contributive_node_idx)
            self.set_leaf(df, less_contributive_node_idx)
            #candidate_children = df[(df.parent_idx.isin(candidate_nodes.node_idx)) & (df.active==1)]
            #self.remove_empty_branches()
            t2 = t.copy()
            t2.df = df.copy()
            self.add_tree(t2)
            iteration_num += 1
            logdata = (iteration_num,
                       len(df[df.active == 1]),
                       less_contributive_node_idx)
            lstr = "Iteration: %s ; leaves: %s; pruned node_idx: %s" % logdata
            logging.debug(lstr)
            if (max_trees is not None) and \
               (len(self.context_trees) > max_trees):
                break
        return self

    def remove_leaves(self, df, node_idx):
        df.loc[df.parent_idx == node_idx, 'active'] = 0
        df.loc[df.node_idx == node_idx, 'num_total_leaves'] = 0
        df.loc[df.node_idx == node_idx, 'num_direct_leaves'] = 0

    def set_leaf(self, df, node_idx):
        try:
            parent = df.loc[df[df.node_idx == node_idx].parent_idx].iloc[0]
        except ValueError:
            df.loc[df.node_idx == node_idx, 'active'] = 1
            return
        if parent.num_child_nodes == 1:
            df.loc[df.node_idx == parent.node_idx, 'num_total_leaves'] = 0
            df.loc[df.node_idx == parent.node_idx, 'num_direct_leaves'] = 0
            df.loc[df.node_idx == node_idx, 'active'] = 0
            self.set_leaf(df, parent.node_idx)
        else:
            df.loc[df.node_idx == node_idx, 'active'] = 1


