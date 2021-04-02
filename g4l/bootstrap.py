import os
import warnings
import pandas as pd
from .util.caching.smc import hashstr
from .resampling import BlockResampling
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


class Bootstrap():

    def __init__(self, cache_folder, num_resamples,
                 renewal_point):
        self.cache_folder = cache_folder
        self.num_resamples = num_resamples
        self.renewal_point = renewal_point

    def get_resamples(self, X, resample_size, num_cores=0):
        # Generate samples using block resampling strategy
        filename = self._resamples_filename(X, resample_size)
        if os.path.isfile(filename):
            return filename
        resample_fctry = BlockResampling(X,
                                         filename,
                                         resample_size,
                                         self.renewal_point)
        resample_fctry.generate(self.num_resamples, num_cores=num_cores)
        return filename

    def _resamples_filename(self, X, size):
        smpl_hash = hashstr(X.data)
        filename = '%s_%s.txt' % (int(size), int(self.num_resamples))
        return os.path.join(self.cache_folder,
                            'samples', smpl_hash,
                            'resamples', filename)
