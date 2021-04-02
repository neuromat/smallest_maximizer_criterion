def g4l(t):
    """
    The paper considers the degrees of freedom as
    the number of possible transitions from a node to other symbols
    """
    d = t.transition_probs
    num_nodes = d[d.freq > 0].groupby(['idx']).count().next_symbol
    return -num_nodes


def csizar_and_talata(t):
    return ((1 - len(t.sample.A)) / 2)


def original_perl(t):
    """
    The perl version of this algorithm uses df = |A|-1
    perl: $LPS = $LPMLS - log($LD) * ( ( $LA - 1 ) * $Pena );
    """
    return -(len(t.sample.A)-1)


def degrees_of_freedom(method_name, t):
    return {'g4l': g4l,
            'perl': original_perl,
            'csizar_and_talata': csizar_and_talata
            }[method_name](t)
