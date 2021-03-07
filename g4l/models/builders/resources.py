import numpy as np


def calculate_num_child_nodes(df):
    """
    Given a context tree dataframe, this method creates a column with
    the number of (immediate) children
    """
    df['active_children'] = 0
    num_child_nodes = (df[df.depth >= 1]
                       .reset_index(drop=False)
                       .groupby(['parent_idx'])
                       .count()).node_idx

    active_children = (df[(df.depth >= 1) & (df.active == 1)]
                       .reset_index(drop=False)
                       .groupby(['parent_idx'])
                       .sum()).active

##    num_child_nodes = (df[df.depth >= 1].reset_index(drop=False).groupby(['parent_idx'])
##                       .reset_index(drop=False)
##                       .groupby(['parent_idx'])
##                       .apply(lambda x: x.count().node_idx))
##
#    active_children = (df[df.depth >= 1]
#                       .reset_index(drop=False)
#                       .groupby(['parent_idx'])
#                       .apply(lambda x: x.sum().active))




    if df.index.name != 'node_idx':
        df.set_index('node_idx', inplace=True)
    #try:
    df['num_child_nodes'] = num_child_nodes
    df['active_children'] = active_children
        #df['active_children'] = df['active_children'].fillna(0)
    #except ValueError:

        #df['num_child_nodes'] = 0

    df.reset_index(inplace=True, drop=False)
    return df


def bind_parent_nodes(df):
    """
    Connects nodes to their parents through the 'parent_idx' column
    """
    pa = df.node.str.slice(start=1)
    parent_idxs = df.reset_index().set_index('node', drop=False).loc[pa.values].node_idx
    #df.set_index('node_idx', inplace=True)
    df['parent_idx'] = parent_idxs.values
    df.loc[df.node == '', 'parent_idx'] = -1
    #df.set_index('node_idx', inplace=True)
    return df
