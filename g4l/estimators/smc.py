from g4l.estimators.base import CollectionBase
from g4l.estimators.logics import smc as smc


class SMC(CollectionBase):
    """
    Context tree selection using the Smallest Maximizer Criterion
    Galves, A., Galves, C., García, J. E., Garcia, N. L., & Leonardi, F. (2010).
    Context tree selection and linguistic rhythm retrieval from written texts.
    Annals of Applied Statistics, 4(1), 186–209.
    https://doi.org/10.1214/11-AOAS511

    Download available at https://arxiv.org/abs/0902.3619

    # adicionar contato

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
                 epsilon=0.01, cache_dir=None, callback_fn=None):
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

        """
        assert max_depth > 0, 'max depth must be greater than zero'
        assert epsilon > 0, 'epsilon must be greater than zero'
        super().__init__()
        self.max_depth = max_depth
        self.penalty_interval = penalty_interval
        self.epsilon = epsilon
        self.cache_dir = cache_dir
        self.callback_fn = callback_fn
        self.tresholds = []

    def fit(self, X):
        """
        Parameters
        ----------
        X : g4l.data.Sample
            A sample object

        ## informar dados da saída

        """
        return smc.fit(self, X)
