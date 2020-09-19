from collections import defaultdict

class Compressor():
    def __init__(self, t):
        self.nodes = t.tree().sort_values('freq', ascending=False)[['node', 'node_idx']].set_index('node')
        self.max_node_len = max(map(lambda x: len(x), self.nodes.index))

    def find_suffix(self, data, i=1):
        if i>self.max_node_len:
            return (None, None, None)
        node = data[-i:]
        return self.find_node(node) or self.find_suffix(data, i+1)

    def find_node(self, node):
        try:
            return (self.nodes.loc[node].node_idx, node, len(node))
        except:
            return None

#    def segmentation(self, X):
#        dic = defaultdict(lambda: 0)
#        d = X.data
#        while len(d)>0:
#            r = find_suffix(d, max_depth)
#            dic[r[1]]+=1
#            d = d[:-r[2]]
#        f = pd.DataFrame([[k, v] for k, v in dic.items()], columns=['node', 'freq']).sort_values(['node'])
#        return f

    def compress(self, X):
        arr = []
        q = []
        d = X.data
        len_ini = len(d)
        while len(d)>0:
            node_idx, _node, node_len = self.find_suffix(d)
            if node_idx is None:
                #print('nooo', d[-1], len(d), len_ini)
                node_len = 1
                arr.append(None)
                q.append(d[-1])
                #break
            arr.append(node_idx)
            d = d[:-node_len]
        return arr, q
