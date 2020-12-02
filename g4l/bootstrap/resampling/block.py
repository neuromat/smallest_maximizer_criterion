import random
import pandas as pd
import numpy as np
from .base import ResamplingBase
from multiprocessing import Pool
import tqdm

def generate_sample(params):
    np.random.seed()
    (data, file, renewal_point, resample_size) = params
    slices = np.array(np.char.split([data], str(renewal_point)[0]))[0]
    num_slices = len(slices)
    idxs = np.random.randint(num_slices, size=int(resample_size))
    resample = ''.join(['%s%s' % (slices[idx], renewal_point) for idx in idxs])
    with open(file, 'a') as f:
        f.write(resample[:int(resample_size)] + '\n')
    return None


class BlockResampling(ResamplingBase):
    def __init__(self, sample, renewal_point=None):
        self.renewal_point = renewal_point
        self.sample = sample

    def iterate(self, file):
        resamples = open(file).read().split('\n')[:-1]
        for i, resample in enumerate(resamples):
            yield (i, resample)

    def generate(self, resample_size, num_resamples, file, num_cores=3):
        data = self.sample.data
        prms = (data, file, self.renewal_point, resample_size)
        params = [prms for i in range(num_resamples)]
        with Pool(num_cores) as p:
            p.map(generate_sample, params)
        #fn = generate_sample(data, file, self.renewal_point, resample_size)
        #with Pool(num_cores) as p:
        #    p.map(fn, range(num_resamples)
        #    tqdm.tqdm(p.map(fn, range(num_resamples)), total=num_resamples)
            #p.map(fn, range(num_resamples))
        #for b in range(num_resamples):
        #    generate_sample(data, file, self.renewal_point, resample_size)


    def __most_frequent_substring(self, source_sample):
        # TODO: implementation
        return '0'
