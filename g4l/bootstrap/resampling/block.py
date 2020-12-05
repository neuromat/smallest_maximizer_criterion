import random
import pandas as pd
import numpy as np
import os
from .base import ResamplingBase
from multiprocessing import Pool
import tqdm


def generate_sample(params):
    np.random.seed()
    (data, file, renewal_point, resample_size) = params

    # 1. Split the sample using renewal point
    # "000100100010" -> ['000', '00', '000', '0']
    slices = np.array(np.char.split([data], str(renewal_point)[0]))[0]

    # 2. count slices
    num_slices = len(slices)

    # 3. generate many random numbers with range (0, num_slices)
    idxs = np.random.randint(num_slices, size=int(resample_size))

    # 4. generate a new sequence by collecting slices according to the indexes
    #  . add the renewal point when joining subsequences
    resample = ''.join(['%s%s' % (slices[idx], renewal_point) for idx in idxs])
    with open(file, 'a') as f:
        # truncate the sequence to match the resample_size
        f.write(resample[:int(resample_size)] + '\n')
    return None


class BlockResampling(ResamplingBase):

    def __init__(self, sample, file, resample_sizes, renewal_point):
        self.sample = sample
        self.file = file
        self.resample_sizes = resample_sizes
        self.renewal_point = renewal_point

    def iterate(self, file):
        resamples = open(file).read().split('\n')[:-1]
        for i, resample in enumerate(resamples):
            yield (i, resample)

    def generate(self, num_resamples, num_cores=3):
        data = self.sample.data
        os.makedirs(os.path.dirname(self.file), exist_ok=True)
        with open(self.file, 'w') as f:
            f.write('')
        prms = (data, self.file, self.renewal_point, max(self.resample_sizes))
        params = [prms for i in range(num_resamples)]
        if num_cores is None:
            for p in params:
                generate_sample(p)
        else:
            with Pool(num_cores) as p:
                p.map(generate_sample, params)
