import numpy as np
import os
import pandas as pd
import logging
from collections import defaultdict


class Sample():
    filename = None
    A = None
    indexes = []

    def __init__(self, filename, A, max_depth,
                 data=None,
                 cache_file=None,
                 subsamples_separator=None):
        self.cache_file = cache_file
        self.max_depth = max_depth
        self.F = None
        self.subsamples_separator = subsamples_separator
        self._set_filename(filename)
        self._load_data(data)
        self._set_A(A)
        self.data_len = len([x for x in self.data if x in self.A])
        self._calc_freqs_and_transitions()


    def _load_data(self, data):
        if data is None:
            self.data = self._load()
        else:
            self.data = data.rstrip()

    def _set_A(self, A):
        if A is not None:
            self.A = [str(a) for a in A]
        else:
            self.A = sorted(list(np.unique([c for c in self.data if c != self.subsamples_separator ])))

    def _set_filename(self, filename):
        try:
            self.filename = os.path.abspath(filename)
        except:
            pass

    def _calc_freqs_and_transitions(self):
        if self.F is None:
            if self.cache_file is None:
                self.F = self._calculate_substr_frequencies()
            else:
                self.F = self.load_cache()
                if self.F is None:
                    self.F = self.save_cache()
        return self.F

    def _calculate_substr_frequencies(self):
        d = defaultdict(lambda: np.zeros(len(self.A)))
        max_depth = self.max_depth
        for smpl in self.subsamples():
            dt = smpl.data
            for i in range(len(dt) - max_depth):
                arr = dt[i: i + max_depth + 1]
                for j in range(len(arr)-1):
                    d[arr[j:-1]][self.A.index(arr[-1])] += 1
        d[''] = [sum(d[a]) for a in self.A]
        return pd.DataFrame.from_dict(d).T

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

    def load_cache(self):
        try:
            f = pd.read_pickle(self.cache_file).astype(int)
            logging.debug("Loaded sample freqs from cache")
            return f
        except FileNotFoundError:
            return None

    def len(self):
        return self.data_len

    def to_a(self):
        if len(self.indexes) == 0:
            self.indexes = [self.A.index(i) for i in self.data]
        return self.indexes

    def data_array(self):
        if self.separator is None:
            return list(self.data)
        else:
            return self.data.split(self.separator)

    def subsamples(self):
        sep = self.subsamples_separator
        if sep is None:
            return [self]
        spl_dt = self.data.split(sep)
        return [Sample(None, self.A, self.max_depth, data=dt) for dt in spl_dt]
