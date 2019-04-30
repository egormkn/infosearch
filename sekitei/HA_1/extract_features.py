# coding: utf-8

import re
import random
import collections
import itertools
import urllib
import urlparse


def extract_features(INPUT_FILE_1, INPUT_FILE_2, OUTPUT_FILE):
    features = collections.defaultdict(int)

    urls_examined = []
    urls_general = []

    with open(INPUT_FILE_1, "r") as file_examined:
        for line in file_examined:
            urls_examined.append(line[:-1])
    file_examined.close()

    with open(INPUT_FILE_2, "r") as file_general:
        for line in file_general:
            urls_general.append(line[:-1])
    file_general.close()

    random.shuffle(urls_examined)
    random.shuffle(urls_general)

    min_size = min(1000, len(urls_examined), len(urls_general))
    del urls_examined[min_size:]
    del urls_general[min_size:]

    n = len(urls_general) + len(urls_examined)
    alpha = 100.0 / n

    for url in itertools.chain(urls_examined, urls_general):
        (scheme, location, path, query, fragment) = urlparse.urlsplit(url)
        path = map(urllib.unquote, path.strip('/').split('/'))
        query = map(lambda kv: map(urllib.unquote, kv.split('=')), query.split('&'))

        features["segments:{len}".format(len=len(path))] += 1
        for (i, parameter) in enumerate(query):
            if len(parameter) > 0 and parameter[0]:
                features["param_name:{name}".format(name=parameter[0])] += 1
            if len(parameter) > 1 and parameter[1]:
                features["param:{name}={value}".format(name=parameter[0], value=parameter[1])] += 1
        for (i, segment) in enumerate(path):
            substr = re.match(r"(\D+\d+\D*|\D*\d+\D+)$", segment)
            ext = re.match(r".*\.([a-zA-Z]+)$", segment)
            if ext is not None:
                ext = ext.group(1)
            features["segment_name_{index}:{string}".format(index=i, string=segment)] += 1
            if re.match(r"[0-9]+$", segment):
                features["segment_[0-9]_{index}:1".format(index=i)] += 1
            if substr is not None:
                features["segment_substr[0-9]_{index}:1".format(index=i)] += 1
            if ext is not None:
                features["segment_ext_{index}:{ext}".format(index=i, ext=ext)] += 1
            if substr is not None and ext is not None:
                features["segment_ext_substr[0-9]_{index}:{ext}".format(index=i, ext=ext)] += 1
            features["segment_len_{index}:{len}".format(index=i, len=len(segment))] += 1

    features_list = features.items()
    with open(OUTPUT_FILE, "w") as file_output:
        for (key, value) in sorted(features_list, key=lambda f: f[1], reverse=True):
            if value <= alpha * n:
                break
            file_output.write(key + '\t' + str(value) + '\n')
    file_output.close()
