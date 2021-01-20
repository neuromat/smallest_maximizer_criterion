import pandas as pd
import numpy as np
from . import resources as rsc
from collections import defaultdict


class ContextTreeBuilder:

    def __init__(self, A):
        self.A = A
        self.contexts = []

    def add_context(self, context, transition_freqs):
        self.contexts.append((context, transition_freqs))

    def build(self):
        from .. import ContextTree
        # import code; code.interact(local=dict(globals(), **locals()))
        if len([c for c, f in self.contexts if c == '']) == 0:
            empty_tr_freq_dic = dict([(ctx, sum(ps)) for ctx, ps in self.contexts if len(ctx)==1])
            empty_tr_freq_dic = defaultdict(int, empty_tr_freq_dic)
            empty_tr_freq = [empty_tr_freq_dic[x] for x in self.A]
            self.contexts.append(('', empty_tr_freq))
        df = self._build_contexts_dataframe()
        probs = self._build_transition_probs()
        max_depth = df.node.str.len().max()
        return ContextTree(max_depth, df, probs)

    def _build_transition_probs(self):
        df = pd.DataFrame(columns=['idx', 'next_symbol', 'freq', 'prob'])
        for i, context in enumerate(self.contexts):
            for a_i, a in enumerate(self.A):
                freqs = context[1]
                row = [i, a, int(freqs[a_i]), freqs[a_i]/sum(freqs)]
                df.loc[len(df)] = row
        df.set_index(['idx'], inplace=True)
        df = df[df.freq > 0]
        return df

    def _build_contexts_dataframe(self):
        df = pd.DataFrame(columns=['node_idx', 'node', 'freq'])
        #max_depth, contexts_dataframe, transition_probs, source_sample=None
        self.contexts.sort(key=lambda x: x[0])
        for i, (node, freq) in enumerate(self.contexts):
            df.loc[len(df)] = [i, node, sum(freq)]

        df['active'] = 1
        df['depth'] = df.node.str.len()
        df = self._build_parents(df)
        df.sort_values(['depth'], inplace=True)
        df.reset_index(drop=False, inplace=True)
        rsc.bind_parent_nodes(df)
        rsc.calculate_num_child_nodes(df)
        df.reset_index(inplace=True, drop=True)
        return df

    def _build_parents(self, df):
        m = df.node.str.len().max()
        internal_nodes = [df.node.str.slice(start=-v).values for v in range(m)]
        internal_nodes = np.unique(np.hstack(internal_nodes))
        internal_nodes = [i for i in internal_nodes if len(i) > 0]
        internal_nodes = [i for i in internal_nodes if i not in df.node.values]
        df.set_index(['node_idx'], inplace=True)
        for n in internal_nodes:
            freqs_sum = df[df.node.str.slice(start=-len(n)) == n].freq.sum()
            df.loc[len(df)] = [n, int(freqs_sum), 0, len(n)]
        return df
