import os
import warnings
import pandas as pd
from g4l.util import hashstr
from g4l.bootstrap.resampling import BlockResampling
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


class Bootstrap():

    def __init__(self, X, cache_folder, size, num_resamples,
                 renewal_point, num_cores=0):
        self.X = X
        self.cache_folder = cache_folder
        self.size = size
        self.num_resamples = num_resamples
        self.renewal_point = renewal_point
        self.num_cores = num_cores

    def _resamples_filename(self):
        #1/0
        smpl_hash = hashstr(self.X.data)
        filename = '%s_%s.txt' % (int(self.size), int(self.num_resamples))
        return os.path.join(self.cache_folder,
                            'samples', smpl_hash,
                            'resamples', filename)

    def resamples(self):
        # Generate samples using block resampling strategy
        filename = self._resamples_filename()
        if os.path.isfile(filename):
            return filename
        resample_fctry = BlockResampling(self.X,
                                         filename,
                                         self.size,
                                         self.renewal_point)
        resample_fctry.generate(self.num_resamples, num_cores=self.num_cores)
        return filename

    # def _initialize_diffs(self, num_trees, num_resamples):
    #     m = np.zeros((num_trees-1, num_resamples))
    #     return (m, m.copy())

