from g4l.data import persistence
import math


class MatSamples:
    def __init__(self, folder, model_name, sample_size, A, max_depth, scan_offset=0):
        self.folder = folder
        self.model_name = model_name
        self.max_depth = max_depth
        self.sample_size = sample_size
        self.scan_offset = scan_offset
        self.A = A
        self.key = '%s_%s' % (model_name, sample_size)
        self.filename = '%s/%s.mat' % (self.folder, self.key)

    def fetch_samples(self, max_samples=math.inf):
        i = -1
        for s in persistence.iterate_from_mat(self.filename, self.key, self.A, self.max_depth, self.scan_offset):
            if i > max_samples:
                break
            i += 1
            yield i, s

    def sample_by_idx(self, idx):
        samples = [x for x in self.fetch_samples(idx)]
        return samples[idx][1]
