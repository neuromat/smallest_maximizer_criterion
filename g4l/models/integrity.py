import logging
import math

def satisfies_properness(df):
  """
  No string in the S is a suffix of any other string in the S
  """

  for depth in reversed(range(2, df.depth.max()+1)):
    for candidate_idx in range(1, depth):
      nodes_to_test = df[df.depth==depth].node.str.slice(candidate_idx).unique()
      unexpected_nodes = df[df.node.isin(nodes_to_test)]
      if len(unexpected_nodes)>0:
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
      df.loc[df.node_idx==row.node_idx, 'node'] = row.node[i:]
      if satisfies_properness(df):
        return False
  return True

def check_admissibility(t, X):
  """
  t is admissible for the sample X1...Xn when the cases below are satisfied:
  """

  ## irreductibility:
  ## when suffix property is satisfied

  assert(satisfies_properness(t.tree()))
  assert(satisfies_irreductibility(t))

  ## when the maximum node length is no more than the specified max_depth $
  max_context_len = t.tree().node.str.len().max()
  assert(max_context_len <= t.max_depth)

  ## when \\sum_{b \\in A}N_{n}(wb) > 0 $
  ## i.e., all contexts in t have transition to at least one symbol
  transitions = t.transition_probs.reset_index().groupby('idx').freq.sum()
  nodes_with_transition_freq = t.tree().set_index('node_idx')[transitions>0]
  assert(len(nodes_with_transition_freq)==len(t.tree()))

  for j in range(t.max_depth, len(X.data)):
    if (j % math.floor(len(X.data)/100))==0:
      tx = 'Checking admissibility: %s ' % round(j/len(X.data)*100, 0)
      logging.debug(tx + "%")
    possible_suffixes = [X.data[j-i-1:j] for i in range(min(j, t.max_depth))]
    assert t.tree().node.isin(possible_suffixes).astype(int).sum()==1 #, msg3 # only one context
    #msg3 = "suffixes for ...%s = %s" % (X.data[t.max_depth:j])
  return True

def satisfies_completeness(t, X):
  """
  For any j = max_depth..(n-1), there is a context in t which
  is suffix of X1..Xj## for any j = max_depth..(n-1), there is a context
  in t which is suffix of X1..Xj
  """
  for j in range(t.max_depth, len(X.data)):
    if (j % math.floor(len(X.data)/100))==0:
      tx = 'Checking completeness: %s ' % round(j/len(X.data)*100, 0)
      logging.debug(tx + "%")
    possible_suffixes = [X.data[j-i-1:j] for i in range(min(j, t.max_depth))]
    return t.tree().node.isin(possible_suffixes).astype(int).sum()==1