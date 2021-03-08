import numpy as np
import os
from os.path import expanduser
import pandas as pd
import logging
from tqdm import tqdm

from collections import defaultdict


class Sample():
    filename = None
    A = None
    indexes = []

    def __init__(self, filename, A, max_depth,
                 data=None,
                 cache_file=None,
                 scan_offset=0,
                 perl_compatible=False,
                 subsamples_separator=None):
        self._set_cache_file(cache_file)
        self.max_depth = max_depth
        self.scan_offset = scan_offset
        self.perl_compatible = perl_compatible
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
        if self.perl_compatible is True:
            return self._compat()
        d = defaultdict(lambda: np.zeros(len(self.A)))
        max_depth = self.max_depth
        logging.debug('Calculating node frequencies in sample')
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

    def _load(self):
        with open(self.filename, 'r') as f:
            txt = f.read()
        return txt.rstrip()

    def save_cache(self):
        logging.debug("Saved sample freqs in cache")
        d = self._calculate_substr_frequencies()
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        #import code; code.interact(local=dict(globals(), **locals()))
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

    def _compat(self):
        from itertools import product
        import re
        def get_substring_count(s, sub_s):
            return sum(1 for m in re.finditer('(?=%s)' % sub_s, s))

        d = defaultdict(lambda: np.zeros(len(self.A)))
        dfreq = defaultdict(lambda: 0)
        for r in range(self.max_depth+1):
            nodes = [''.join(x) for x in list(product(self.A, repeat=r))]
            for node in nodes:

                dfreq[node] = get_substring_count(self.data, node)
                d[node] = [get_substring_count(self.data, node + x) for x in self.A]

        df = pd.DataFrame.from_dict(d).T
        df['N'] = [dfreq[x] for x in list(df.index)]
        return df
