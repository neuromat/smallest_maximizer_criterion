from . import compression
from . import parallel
from . import stats
from hashlib import md5


def unique_chars(file):
    from collections import Counter
    return Counter([x for x in open(file).read()[:-1]])


def hashstr(strg):
    return md5((strg).encode('utf-8')).hexdigest()
