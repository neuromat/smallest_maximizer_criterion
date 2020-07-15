# Following the conventions used by scikit-learn
# https://scikit-learn.org/stable/developers/develop.html
from g4l import SmallestMaximizerCriterion
from g4l.estimators.ctm_scanner import CTMScanner
from g4l.evaluation.bootstrap import Bootstrap
from g4l.evaluation.t_test import TTest
from g4l.data import Sample

# Create a sample object instance
X = Sample('./g4l/fixtures/folha.txt', [0, 1, 2, 3, 4])

# Define the champion trees strategy to be used
#ctm_scan = CTMScanner(penalty_interval=(0.1, 400), epsilon=0.01)
ctm_scan = CTMScanner(penalty_interval=(0.1, 2), epsilon=0.5)


# Instantiates SMC by passing the strategies that will
# be used to generate the candidate trees
scm = SmallestMaximizerCriterion(ctm_scan, max_depth=4)


# Define the champion trees strategy to be used
num_resamples = 200
bootstrap = Bootstrap(X, partition_string='4')
small_resamples = bootstrap.resample(num_resamples, size=len(X.data) * 0.3)
large_resamples = bootstrap.resample(num_resamples, size=len(X.data) * 0.9)
t_test = TTest(small_resamples, large_resamples, alpha=0.01)


# Run estimator
scm.fit(X, t_test, processors=3)
import code; code.interact(local=dict(globals(), **locals()))

# Returns the best tree
scm.best_tree.to_str()

# Evaluates a new sample with the model with previously fitted params
scm.score(Xb)


# optional one-liner call
scm.fit(X, max_depth=4).score(Xb)
