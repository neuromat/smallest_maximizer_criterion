def find_renewal_points(context_tree, sample):
    ctx_trans = transition_context2context(context_tree, sample)

    import code; code.interact(local=dict(globals(), **locals()))
    renewals = []
    pass


def transition_context2context(context_tree, sample):
    df = context_tree.transition_probs.set_index('idx')
    t = context_tree.tree()
    tw = t[['node_idx', 'node']].set_index('node_idx').join(df[df.prob>0])
    tw['next_node'] = tw[['node', 'next_symbol']].agg(''.join, axis=1)
    for _, node in t.iterrows():
        tw.loc[tw.next_node.str.endswith(node.node), 'next_context_idx'] = node.node_idx
    return tw

def is_renewal_point():
    pass
