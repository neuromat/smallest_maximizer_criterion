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
                       .apply(lambda x: x.count().node_idx))

    active_children = (df[df.depth >= 1]
                       .reset_index(drop=False)
                       .groupby(['parent_idx'])
                       .apply(lambda x: x.sum().active))


    if df.index.name != 'node_idx':
        df.set_index('node_idx', inplace=True)
    try:
        df['num_child_nodes'] = num_child_nodes
        df['active_children'] = active_children
        df['active_children'] = df['active_children'].fillna(0)
    except ValueError:
        df['num_child_nodes'] = 0

    df.reset_index(inplace=True, drop=False)
    return df


def bind_parent_nodes(df):
    """
    Connects nodes to their parents through the 'parent_idx' column
    """
    empty_node_idx = df[df.node == ''].index
    df['parent_node'] = df.node.str.slice(start=1)
    df.reset_index(inplace=True)
    df.set_index('node', inplace=True)
    parent_nodes = df[df.depth > 1].parent_node
    parent_nodes_idx = df.loc[parent_nodes].node_idx
    df.reset_index(inplace=True)
    parent_nodes_idx = parent_nodes_idx.reset_index().node_idx.values
    #depth_1_values = np.repeat(None, len(df.loc[df.depth == 1]))
    depth_0_values = [None]
    depth_1_values = np.repeat(empty_node_idx, len(df.loc[df.depth == 1]))
    df['parent_idx'] = np.concatenate((depth_0_values, depth_1_values, parent_nodes_idx))
    df.drop('parent_node', axis='columns', inplace=True)
    df.set_index('node_idx', inplace=True)
    return df
