import numpy as np
import os
from multiprocessing import Pool


def generate_sample(params):
    """
    Generates a sample using block strategy
    This method is executed in parallel for many resample instances
    """
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


class BlockResampling():

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
