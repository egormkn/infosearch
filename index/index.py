# coding: utf-8

import cPickle as pickle
import collections
import argparse
import docreader
import doc2words
import varbyte
import simple9

from mmh3 import hash


parser = argparse.ArgumentParser(description="Index builder")
parser.add_argument("method", help="Compression method")
parser.add_argument("files", nargs="+", help="Input files (.gz or plain) to process")
args = parser.parse_args()

if args.method == "varbyte":
    encoder = varbyte
elif args.method == "simple9":
    encoder = simple9
else:
    raise AssertionError("Method {name} is not supported".format(name=args.method))

inverted_index = collections.defaultdict(list)
urls = []
reader = docreader.DocumentStreamReader(args.files)

for (doc_id, doc) in enumerate(reader):
    urls.append(doc.url)
    words = doc2words.extract_words(doc.text)
    for word in set(words):
        word = word.encode("utf-8")
        inverted_index[hash(word)].append(doc_id)

for key in inverted_index:
    inverted_index[key] = encoder.encode(inverted_index[key])

with open("index", "w") as file_index:
    pickle.dump(args.method, file_index)
    pickle.dump(inverted_index, file_index)
file_index.close()

with open("urls", "w") as file_urls:
    pickle.dump(urls, file_urls)
file_urls.close()
