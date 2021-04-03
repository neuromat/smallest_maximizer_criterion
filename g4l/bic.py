from .smc_base import EstimatorsBase
from .estimators.bic import *
from .context_tree import ContextTree
from .util import degrees_of_freedom as dof


class BIC(EstimatorsBase):
    """
    Implements the BIC tree estimator described in:
    Context tree selection using the Smallest Maximizer Criterion
    Galves, A., Galves, C., García, J. E., Garcia, N. L., & Leonardi, F. (2010).
    Context tree selection and linguistic rhythm retrieval from written texts.
    Annals of Applied Statistics, 4(1), 186–209.
    https://doi.org/10.1214/11-AOAS511

    Download available at https://arxiv.org/abs/0902.3619

    Originally presented by Csiszár and Talata (2006)
    in the paper 'Context tree estimation for not necessarily finite memory
    processes, Via BIC and MDL'

    ...

    Attributes
    ----------
    context_tree : list
        Champion tree found by the estimator

    Methods
    -------
    fit(X)
        Estimates champion trees for the given sample X and constant c
    """

    def __init__(self, c, df_method='perl', keep_data=False, perl_compatible=False):
        """
        Parameters
        ----------
        c : tuple
            The penalization constant used in the BIC estimator
        df_method : str
            The method used to calculate degrees_of_freedom. Options:
            - 'perl': uses the same df as the original implementation in perl
            - 'g4l': uses the method as described in the paper (slightly different)
            - 'ct06': uses df as described in Csizar and Talata (2006)
            (check g4l.util.degrees_of_freedom for further details)
        keep_data : bool
            When set to True, tree files are generated with all internal nodes
            (for inspection/analysis purposes)
        perl_compatible : bool
            Set algorithm to be compatible with the paper's perl version
        """
        assert c >= 0, 'c must be at least zero'
        self.c = c
        self.df_method = df_method
        self.keep_data = keep_data
        self.perl_compatible = perl_compatible

    def fit(self, X):
        """ Estimates Context Tree model using BIC """

        full_tree = ContextTree.init_from_sample(X, force_admissible=False)
        df, transition_probs = full_tree.df, full_tree.transition_probs

        deg_f = dof.degrees_of_freedom(self.df_method, full_tree)
        penalty = penalty_term(X.len(), self.c, deg_f)
        df['likelihood_pen'] = penalty + df.likelihood
        full_tree.df = assign_values(X.max_depth, df[df.freq >= 1],
                                     self.perl_compatible)
        full_tree.prune_unique_context_paths()
        self.context_tree = clean_columns(full_tree, self.keep_data)
        return self
