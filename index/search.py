# coding: utf-8

import cPickle as pickle
import sys
import varbyte
import simple9

from mmh3 import hash


def intersect(x, y):
    (i, j) = (0, 0)
    intersection = []
    while i < len(x) and j < len(y):
        if x[i] < y[j]:
            i += 1
        elif y[j] < x[i]:
            j += 1
        else:
            intersection.append(y[j])
            j += 1
            i += 1
    return intersection


with open("index", "r") as file_index:
    method = pickle.load(file_index)
    inverted_index = pickle.load(file_index)
file_index.close()

with open("urls", "r") as file_urls:
    urls = pickle.load(file_urls)
file_urls.close()

if method == "varbyte":
    encoder = varbyte
elif method == "simple9":
    encoder = simple9
else:
    raise AssertionError("Method {name} is not supported".format(name=method))

for line in sys.stdin:
    result = []
    for (i, word) in enumerate(line.decode('utf-8').split('&')):
        word = word.strip().lower().encode('utf-8')
        docs = encoder.decode(inverted_index[hash(word)])
        if i == 0:
            result = docs
        else:
            result = intersect(result, docs)
    print line
    print len(result)
    for doc_id in result:
        print urls[doc_id]
