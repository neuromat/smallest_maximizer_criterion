
import matplotlib.pyplot as plt

def bloxplot(smc_instance):
    plt.figure(figsize=(24, 18))
    df = smc_instance.df
    for i, t in enumerate(smc_instance.champion_trees):
        qlml1 = df[(df.tree_idx==i) & (df.sample_size=='sm')].lml_delta_div_n
        qlml2 = df[(df.tree_idx==i) & (df.sample_size=='lg')].lml_delta_div_n
        plt.subplot(4, np.ceil(len(champion_trees)/4), i+1)
        plt.title('$t_{%s}$ - %s'% (i, len(t.to_str().split(' '))))
        plt.boxplot([qlml1, qlml2])

def log_likelihood_per_leaves(smc_instances, labels):
    plt.figure(figsize=(14, 6))
    for idx, instance in enumerate(smc_instances):
        ll = [x.log_likelihood() for x in instance.champion_trees]
        num_contexts = [len(x.leaves()) for x in instance.champion_trees]
        label = labels[idx]
        plt.plot(num_contexts, ll, marker='o', linewidth=1, label=label)
    plt.title("BP/EP Log-likelihood functions")
    plt.grid()
    plt.legend()


def toytree(tree, layout='l', width=300, height=350):
    import toytree
    tt = tree.to_ete()
    tre0 = toytree.tree(tt.write(format=8))
    tre0.draw(
        layout=layout,
        tree_style='c',
        #node_labels=True,
        tip_labels=True,
        tip_labels_align=False,
        node_sizes=[6 if i else 0 for i in tre0.get_node_values()],
        node_style={"stroke": "black", "fill": "black"},
        node_labels_style={
            "fill": "#262626",
            "font-size": "10px",
            "-toyplot-anchor-shift": "12px",
        },
        scalebar=False,
        width=width,
        height=height,
    )


def draw_tree(tree, size='10,10', previous_tree=None, diff_color='black'):
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
