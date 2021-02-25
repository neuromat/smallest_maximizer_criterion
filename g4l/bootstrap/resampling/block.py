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
    renewal_point = str(renewal_point)
    resample_size = int(resample_size)
    X = data
    lrp = len(str(renewal_point))
    idx = np.zeros(resample_size)
    nrenewals = 1
    for i in range(resample_size - lrp):
        if X[i:i+lrp] == renewal_point:
            idx[nrenewals] = i
            nrenewals += 1
    idx = idx[:nrenewals].astype(int)
    l_idx = len(idx)
    #blocks = cell(l_idx - 1, 1)
    #import code; code.interact(local=dict(globals(), **locals()))
    blocks = np.empty(l_idx - 1, dtype='object')
    for i in range(l_idx-1):
        blocks[i] = X[idx[i]: idx[i+1]]

    #  create the bootstrap samples
    nblocks = len(blocks)
    lblocks = list(map(lambda x: len(x), blocks))

    B = np.empty(resample_size + max(lblocks), dtype='object')
    nseq = -1
    it = 0
    while (nseq < resample_size):
        blk = np.random.randint(nblocks)
        B[it] = blocks[blk]
        it += 1
        nseq = nseq + lblocks[blk]
    ret = ''.join(B[:it])[:int(resample_size)]
    with open(file, 'a') as f:
        f.write(ret + '\n')
    #import code; code.interact(local=dict(globals(), **locals()))
    #pass

# TODO: escrever teste comparando os métodos usando o mesmo seed
def generate_sample2(params):
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
    resample = ''.join(['%s%s' % (renewal_point, slices[idx]) for idx in idxs])
    with open(file, 'a') as f:
        # truncate the sequence to match the resample_size
        f.write(resample[:int(resample_size)] + '\n')
    return None


class BlockResampling(ResamplingBase):

    def __init__(self, sample, file, resample_size, renewal_point):
        self.sample = sample
        self.file = file
        self.resample_size = resample_size
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
        prms = (data, self.file, self.renewal_point, self.resample_size)
        params = [prms for i in range(num_resamples)]
        if num_cores in [None, 0, 1]:
            for p in params:
                generate_sample(p)
        else:
            with Pool(num_cores) as p:
                p.map(generate_sample, params)
