def g4l(t):
    """
    The paper considers the degrees of freedom as
    the number of possible transitions from a node to other symbols
    """
    d = t.transition_probs
    num_nodes = d[d.freq > 0].groupby(['idx']).count().next_symbol
    return -num_nodes


def ct06(t):
    """
    This method implements df
    as described in Csiszar and Talata (2006)
    DOI: 10.1109/TIT.2005.864431
    """
    return ((1 - len(t.sample.A)) / 2)


def original_perl(t):
    """
    The perl version of this algorithm uses df = |A|-1
    perl code: $LPS = $LPMLS - log($LD) * ( ( $LA - 1 ) * $Pena );
    """
    return -(len(t.sample.A)-1)


def degrees_of_freedom(method_name, tree):
    """
    Returns the calculated df given one of the three
    different strategies available
    """
    return {'g4l': g4l,
            'perl': original_perl,
            'ct06': ct06
            }[method_name](tree)
