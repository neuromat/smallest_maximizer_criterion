from graphviz import Digraph
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

def draw_tree(tree):
    if isinstance(tree, str):
      t = tree
    else:
      t = tree.to_str()
    dot = Digraph(engine='dot', name=t)
    dot.attr(size='10,10')
    dot.attr('node', shape='circle')
    contexts = t.split(' ')
    d = []
    dot.node(name='$', label='root')
    for i in range(len(contexts)):
        ctx = contexts[i][::-1]
        ctxl = [(ctx[:i+1], c) for i, c in enumerate(ctx)]
        for node in ctxl:
            dot.node(name=node[0], label=node[1])
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
