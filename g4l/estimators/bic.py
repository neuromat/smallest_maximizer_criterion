from .base import Base
from .logics import bic


class BIC(Base):
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

    def __init__(self, c, max_depth, df_method='perl', scan_offset=0, keep_data=False, perl_compatible=False):
        """
        Parameters
        ----------
        max_depth : int
            The maximum depth allowed for the context trees
        c : tuple
            The penalization constant used in the BIC estimator
        df_method : str
            The method used to calculate degrees_of_freedom. Options:
            - 'perl': uses the same df as the original implementation in perl
            - 'g4l': uses the method as described in the paper (slightly different)
            - 'csizar_and_talata': uses df as described in Csizar and Talata (2006)
        scan_offset : int
            Computes node frequencies starting from `scan_offset` position
        perl_compatible : bool
            Set algorithm to be compatible with the paper's perl version
        """
        assert max_depth > 0, 'max depth must be greater than zero'
        assert c >= 0, 'c must be at least zero'
        self.c, self.max_depth = c, max_depth
        self.df_method = df_method
        self.scan_offset = scan_offset
        self.keep_data = keep_data
        self.perl_compatible = perl_compatible

    def fit(self, X):
        """
        Parameters
        ----------
        X : g4l.data.Sample
            A sample object

        ## informar dados da saída

        """
        self.context_tree = bic.fit(X, self.c,
                                    self.max_depth,
                                    self.df_method,
                                    self.scan_offset,
                                    self.perl_compatible,
                                    keep_data=self.keep_data)
        return self
