import logging
import os
from os.path import expanduser
from .estimators import pl_compat as pl
import numpy as np
import pandas as pd
from collections import defaultdict


class Sample():
    filename = None
    A = None
    indexes = []

    def __init__(self, filename, A, max_depth,
                 data=None,
                 cache_file=None,
                 perl_compatible=False,
                 subsamples_separator=None):
        self._set_cache_file(cache_file)
        self.max_depth = max_depth
        self.perl_compatible = perl_compatible
        self.F = None
        self.subsamples_separator = subsamples_separator
        self._set_filename(filename)
        self._load_data(data)
        self._set_A(A)
        self.data_len = len([x for x in self.data if x in self.A])
        self.calc_node_frequencies()

    def calc_node_frequencies(self):
        """Computes the frequencies of nodes in the sample

        This method computes the frequencies of all nodes of length
        up to max_depth in the sample of length. Whenever the cache file
        is specified, it's returned instead of performing calculations
        another time (or created if it still doesn't exist)

        Returns:
            DataFrame -- A table with the node freqs
        """

        if self.F is None:
            if self.cache_file is None:
                self.F = self._calculate_substr_frequencies()

            else:
                self.F = self.load_cache()
                if self.F is None:
                    self.F = self.save_cache()
        return self.F

    def _calculate_substr_frequencies(self):
        """Computes the frequencies of all sequences of length
        up to max_depth in the sample. Each column represents
        a symbol in A, and contains the number of transitions
        from the sequence in the row. The column N represents
        how many times the sequence appears in the sample.

        Example:
                      0       1      N
        01         8405       0   8405
        1          8405       0   8405
        001010      503    1258   1761
        01010      1761    4139   5900
        1010       1761    4139   5900
        010        2506    5899   840

        Returns:
            DataFrame -- A table with the node freqs
        """
        logging.debug('Calculating node frequencies in sample')
        if self.perl_compatible is True:
            return pl.calculate_substr_frequencies(self)
        d = defaultdict(lambda: np.zeros(len(self.A)))
        max_depth = self.max_depth

        for smpl in self.subsamples():
            dt = smpl.data
            ln = len(dt) - max_depth
            for i in range(ln):
                arr = dt[i: i + max_depth + 1]
                for j in range(len(arr)-1):
                    d[arr[j:-1]][self.A.index(arr[-1])] += 1
        d[''] = [sum(d[a]) for a in self.A]
        df = pd.DataFrame.from_dict(d).T
        df['N'] = df.T.sum().astype(int)
        return df

    def _load_data(self, data):
        if data is None:
            self.data = self._load()
        else:
            self.data = data.rstrip()

    def _set_A(self, A):
        if A is not None:
            self.A = [str(a) for a in A]
        else:
            self.A = sorted(list(np.unique([c for c in self.data if c not in [self.subsamples_separator]])))

    def _set_filename(self, filename):
        try:
            self.filename = os.path.abspath(filename)
        except:
            pass

    def _load(self):
        with open(self.filename, 'r') as f:
            txt = f.read()
        return txt.rstrip()

    def save_cache(self):
        logging.debug("Saved sample freqs in cache")
        d = self._calculate_substr_frequencies()
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        d.to_pickle(self.cache_file)
        return self.load_cache()

    def _set_cache_file(self, cache_file):
        if cache_file is None:
            self.cache_file = None
        else:
            self.cache_file = os.path.abspath(expanduser(cache_file))

    def load_cache(self):
        try:
            f = pd.read_pickle(self.cache_file).astype(int)
            logging.debug("Loaded sample freqs from cache")
            return f
        except FileNotFoundError:
            return None

    def len(self):
        return self.data_len

    def subsamples(self):
        """Divides sample using the subsample_separator symbol.
        Returns an array with all sample slices
        or with the sample itself if separator is not defined.

        The linguistic case study samples use it.
        """
        sep = self.subsamples_separator
        if sep is None:
            return [self]
        spl_dt = self.data.split(sep)
        return [Sample(None, self.A, self.max_depth, data=dt) for dt in spl_dt]
