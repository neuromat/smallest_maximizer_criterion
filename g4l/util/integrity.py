import logging
import math
from tqdm import tqdm


def satisfies_properness(df):
    """
    No string in the S is a suffix of any other string in the S
    """

    for depth in reversed(range(2, df.depth.max()+1)):
        for candidate_idx in range(1, depth):
            nodes_to_test = df[df.depth == depth].node.str.slice(candidate_idx).unique()
            unexpected_nodes = df[df.node.isin(nodes_to_test)]
            if len(unexpected_nodes) > 0:
                return False
    return True


def satisfies_irreductibility(t):
    """
    A string in t can't be replaced by a proper suffix without
    violating the suffix property
    """
    for idx, row in t.tree().iterrows():
        for i in range(1, len(row.node)):
            df = t.tree()
            df.loc[df.node_idx == row.node_idx, 'node'] = row.node[i:]
            if satisfies_properness(df):
                return False
    return True


def check_admissibility(t, X):
    """
    t is admissible for the sample X1...Xn when the cases below are satisfied:
    """

    ## irreductibility:
    ## when suffix property is satisfied

    assert satisfies_properness(t.tree()), "Properness not satisfied"

    assert satisfies_irreductibility(t), "Not irreductible"

    ## when the maximum node length is no more than the specified max_depth $
    max_context_len = t.tree().node.str.len().max()
    assert max_context_len <= t.max_depth, "Tree length exceeds max depth"

    ## when \\sum_{b \\in A}N_{n}(wb) > 0 $
    ## i.e., all contexts in t have transition to at least one symbol
    transitions = t.transition_probs.reset_index().groupby('idx').freq.sum()
    nodes_with_transition_freq = t.tree().set_index('node_idx')[transitions>0]
    assert len(nodes_with_transition_freq) == len(t.tree()), "Nodes without transition"

    for j in tqdm(range(t.max_depth, X.len())):
        if (j % math.floor(X.len()/100)) == 0: # for visualization purposes only
            tx = 'Checking admissibility: %s ' % round(j/X.len()*100, 0)
            logging.debug(tx + "%")
        possible_suffixes = [X.data[j-i-1:j] for i in range(min(j, t.max_depth))]
        assert t.tree().node.isin(possible_suffixes).astype(int).sum() == 1, "Not admissible"
        #msg3 = "suffixes for ...%s = %s" % (X.data[t.max_depth:j])
    return True


def satisfies_completeness(t, X):
    """
    For any j = max_depth..(n-1), there is a context in t which
    is suffix of X1..Xj## for any j = max_depth..(n-1), there is a context
    in t which is suffix of X1..Xj
    """
    for j in tqdm(range(X.max_depth, X.len())):
        possible_suffixes = [X.data[j-i-1:j] for i in range(min(j, X.max_depth))]
        assert t.tree().node.isin(possible_suffixes).astype(int).sum() == 1
    return True

# from g4l.models import integrity
# integrity.is_freq_consistent(df)
# integrity.is_freq_consistent(df)

def is_freq_consistent(context_tree):
    df, tp = context_tree.df, context_tree.transition_probs
    node0 = df[df.depth == 0]
    assert len(node0) == 1, "There are more than one zero-depth nodes!"
    parent_idx = node0.node_idx.values[0]
    _count_recursive(df, tp, parent_idx)
    return True


def _count_recursive(df, tp, parent_idx):
    children = df[df.parent_idx == parent_idx]
    parent_freq = df[df.node_idx == parent_idx].freq.values[0]
    assert parent_freq == tp.loc[parent_idx].freq.sum(), "Transition prob freqs doesnt match"
    assert tp.loc[parent_idx].prob.sum() > 0.999, 'Probabilities are inconsistent for node_idx %s' % parent_idx
    if len(children) > 0:
        children_freq_sum = children.freq.sum()
        assert parent_freq == children_freq_sum, "Sum of children freqs doesnt match with node freq"
        for idx, child in children.iterrows():
            _count_recursive(df, tp, idx)
