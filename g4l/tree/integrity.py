# No string in the S is a suffix of any other string in the S
def satisfies_suffix_property(context_tree):
  df = context_tree.tree()
  for depth in reversed(range(2, df.depth.max()+1)):
    for candidate_idx in range(1, depth):
      nodes_to_test = df[df.depth==depth].node.str.slice(candidate_idx).unique()
      unexpected_nodes = df[df.node.isin(nodes_to_test)]
      if len(unexpected_nodes)>0:
        return False
  return True


def satisfies_completeness(context_tree):
  df = context_tree.tree()
  max_depth = df.depth.max()
  X = context_tree.sample.data
  ret = df.node.map(lambda x: X.endswith(x)).astype(int).sum()
  if ret != 1:
    return False
  return True
