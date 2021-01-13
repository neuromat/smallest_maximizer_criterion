import numpy as np
import os
import logging


class Sample():
    filename = None
    A = None
    indexes = []

    def __init__(self, filename, A, data=None, separator=None):
        try:
            self.filename = os.path.abspath(filename)
        except:
            pass
        self.separator = separator
        if data is None:
            self.data = self.load()
        else:
            self.data = data

        if A is not None:
            data_len = len(self.data)
            self.A = [str(a) for a in A]
            self.data = ''.join(c for c in self.data if c in self.A)
            len_diff = data_len - len(self.data)
            if len_diff > 0:
                logging.warning('Invalid characters were filtered from the provided sample (%s occurrences)' % len_diff)
        else:
            if separator == None:
                self.A = list(np.unique([c for c in self.data]))
            else:
                self.A = np.unique(self.data.split(separator))

    def to_a(self):
        if len(self.indexes) == 0:
            self.indexes = [self.A.index(i) for i in self.data]
        return self.indexes

    def data_array(self):
        if self.separator is None:
            return list(self.data)
        else:
            return self.data.split(self.separator)

    def load(self):
        with open(self.filename, 'r') as f:
            txt = f.read()
        return txt.rstrip()
