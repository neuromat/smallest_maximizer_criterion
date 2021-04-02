import matplotlib.pyplot as plt
import networkx as nx


def log_likelihood_per_leaves(smc_instances, labels):   # pragma: no cover
    plt.figure(figsize=(14, 6))
    for idx, instance in enumerate(smc_instances):
        ll = [x.log_likelihood() for x in instance.champion_trees]
        num_contexts = [len(x.leaves()) for x in instance.champion_trees]
        label = labels[idx]
        plt.plot(num_contexts, ll, marker='o', linewidth=1, label=label)
    plt.title("BP/EP Log-likelihood functions")
    plt.grid()
    plt.legend()


def plot2(tree, column_label='symbol', font_size=10,
          label_offset_x=-14,
          label_offset_y=10,
          horizontal_alignment='right',
          node_size=8,
          ax=None,
          linewidths=1.0,
          node_color='black'):   # pragma: no cover
    df = tree.df.copy()
    s1 = list(map(lambda c: set([c[x:] for x in range(len(c))]), df[(df.active==1)].node.values))
    s2 = set([''])
    for s in s1:
        s2.update(s)
    df = df[df.node.isin(s2)]
    root_idx = df[df.node == ''].node_idx.values[0]
    df['symbol'] = df.node.str[0]
    df = df.sort_values(['depth', 'symbol'])
    df.loc[df.parent_idx == -1, 'parent_idx'] = 0
    G = nx.DiGraph()
    for i, node in df.iterrows():
        G.add_node(node.node_idx)
    for i, node in df.iterrows():
        if node.parent_idx > 0:
            G.add_edge(node.parent_idx, node.node_idx)
    root_idx = df[df.node == ''].node_idx.values[0]
    pos = nx.nx_pydot.pydot_layout(G, prog="dot", root=root_idx)
    nx.draw_networkx(G, pos,
                     node_size=node_size,
                     node_color=node_color,
                     alpha=1,
                     with_labels=False,
                     ax=ax,
                     linewidths=linewidths,
                     arrows=False)
    for i, node in df.iterrows():
        x,y=pos[node.node_idx]
        if node.node_idx != root_idx:
            plt.text(x+label_offset_x, y-label_offset_y,
                     s=node[column_label],
                     fontsize=font_size,
                     horizontalalignment=horizontal_alignment)
    plt.axis("off")


def draw_tree(tree, size='10,10', previous_tree=None, diff_color='black'):  # pragma: no cover
    if isinstance(tree, str):
      t = tree
    else:
      t = tree.to_str()
    if previous_tree is None:
        t2 = t
    else:
        t2 = previous_tree.to_str()

    from graphviz import Digraph
    dot = Digraph(engine='dot', name=t, format='png')
    dot.attr(size=size)
    dot.attr('node', shape='circle')
    contexts = t.split(' ')
    contexts2 = t2.split(' ')
    d = []
    dot.node(name='$', label='root')
    for i in range(len(contexts)):
        ctx = contexts[i][::-1]
        ctxl = [(ctx[:i+1], c) for i, c in enumerate(ctx)]
        for node in ctxl:
            color=None
            if contexts2.count(node[0][::-1])==0:
                color=diff_color
            dot.node(name=node[0], label=node[1], color=color, fontcolor=color)
            if(len(node[0]))==1:
                if ('$', node[0]) not in d:
                    d.append(('$', node[0]))
                    dot.edge('$', node[0])
        for pair in list(zip(ctxl, ctxl[1:])):
            n1 = pair[0]
            n2 = pair[1]
            if (n1, n2) not in d:
                d.append((n1, n2))
                dot.edge(n1[0], n2[0])
    return dot
