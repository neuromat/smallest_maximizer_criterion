
def unique_chars(file):
    from collections import Counter
    return Counter([x for x in open(file).read()[:-1]])


