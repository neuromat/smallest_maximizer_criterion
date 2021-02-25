from g4l.estimators.base import CollectionBase
from g4l.estimators.logics import smc as smc
from tempfile import TemporaryDirectory
import numpy as np
import os
import logging


class SMC(CollectionBase):
    """
    Context tree selection using the Smallest Maximizer Criterion
    Galves, A., Galves, C., García, J. E., Garcia, N. L., & Leonardi, F. (2010).
    Context tree selection and linguistic rhythm retrieval from written texts.
    Annals of Applied Statistics, 4(1), 186–209.
    https://doi.org/10.1214/11-AOAS511

    Download available at https://arxiv.org/abs/0902.3619

    ...

    Attributes
    ----------
    context_trees : list
        All champion trees found by the estimator

    thresholds : list
        The constant values used by BIC estimator to produce the
        context trees in the attribute `context_trees`.


    Methods
    -------
    fit(X)
        Estimates champion trees for the given sample X
    """

    def __init__(self, max_depth, penalty_interval=(0.1, 400),
                 epsilon=0.01, cache_dir=None, callback_fn=None,
                 scan_offset=0, df_method='perl', perl_compatible=False):
        """
        Parameters
        ----------
        max_depth : int
            The maximum depth allowed for the context trees
        penalty interval : tuple
            minimum and maximum values for the penalization constant used in
            the BIC estimator
        epsilon : float
            This value sets an stop condition when scanning for new trees
            between two penalty intervals in the BIC criteria
        cache_dir : str
            When this variable is set with a valid path, the `fit` method will
            work cached; for any set of initial paramaters and sample X, there
            will exist a folder that receives the computed champion trees. In
            any further call using the same arguments, the method will returns
            the cached information.
        callback_fn : function
            Whenever a new tree is found, the estimator yields to this callback
            method (when provided). This might is useful for logging and
            interfacing purposes
        df_method : str
            The method used by BIC to calculate degrees_of_freedom. Options:
            - 'perl': uses the same df as the original implementation in perl
            - 'g4l': uses the method as described in the paper (slightly different)
            - 'csizar_and_talata': uses df as described in Csizar and Talata (2006)
        scan_offset : int
            Computes node frequencies starting from `scan_offset` position
        perl_compatible : int
            Makes algorithm compatible with the paper's perl code

        """
        assert max_depth > 0, 'max depth must be greater than zero'
        assert epsilon > 0, 'epsilon must be greater than zero'
        self.temp_cache = None
        cache_dir = cache_dir or self._tempdir()
        super().__init__(cache_dir)
        self.max_depth = max_depth
        self.penalty_interval = penalty_interval
        self.epsilon = epsilon
        self.df_method = df_method
        self.callback_fn = callback_fn
        self.scan_offset = scan_offset
        self.thresholds = []
        self.perl_compatible = perl_compatible

    def clean(self):
        if self.temp_cache is not None:
            self.temp_cache.cleanup()

    def fit(self, X):
        """
        Parameters
        ----------
        X : g4l.data.Sample
            A sample object

        ## informar dados da saída

        """
        smc.fit(self, X)
        self.X = X
        return self

    def _tempdir(self):
        self.temp_cache = TemporaryDirectory()
        return self.temp_cache.name

