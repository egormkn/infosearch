# coding: utf-8

# Learning Name-finder
# See https://arxiv.org/pdf/cmp-lg/9803003.pdf

from __future__ import division

import gzip
import regex as re
import sys
import xml.etree.ElementTree as XMLTree

from itertools import repeat, izip
from enum import Enum
from collections import defaultdict


class Marker(Enum):
    OTHER = 'O'
    START = '+START+'
    END = '+END+'
    ORGANIZATION = 'C'
    PERSON = 'P'
    LOCATION = 'G'

    @staticmethod
    def list():  # Fixes the iterator ordering bug in enum
        return [
            Marker.OTHER,
            Marker.START,
            Marker.END,
            Marker.ORGANIZATION,
            Marker.PERSON,
            Marker.LOCATION
        ]

    @staticmethod
    def get(index):
        return Marker.list()[index]

    def index(self):
        return Marker.list().index(self)

    def colorize(self, text):
        if self == Marker.START or self == Marker.END:
            raise AssertionError("Tried to colorize text with technical marker {}".format(self))
        return {
            Marker.ORGANIZATION: ur"\u001b[36m{}\u001b[0m",
            Marker.PERSON: ur"\u001b[32m{}\u001b[0m",
            Marker.LOCATION: ur"\u001b[31m{}\u001b[0m"
        }.get(self, ur"{}").format(text)


class Feature(Enum):
    UPPER_CASE = re.compile(ur"^([[:upper:]])+$", re.UNICODE)
    CAMEL_CASE = re.compile(ur"^([[:upper:]]+[[:lower:]]*){2,}$", re.UNICODE)
    UPPER_NAME = re.compile(ur"^([[:upper:]]+\.)+[[:upper:]]*$", re.UNICODE)
    CAMEL_NAME = re.compile(ur"^([[:upper:]]+[[:lower:]]*[.])+[[:upper:]]+[[:lower:]]*$", re.UNICODE)
    UPPER_PUNC = re.compile(ur"^([[:upper:]]+[[:punct:]]+)+$", re.UNICODE)
    CAMEL_PUNC = re.compile(ur"^([[:upper:]]+[[:lower:]]*[[:punct:]]+){2,}$", re.UNICODE)
    FIRST_WORD = re.compile(ur"^\b$", re.UNICODE)  # Never matches
    CAPITALIZE = re.compile(ur"^[[:upper:]].*$", re.UNICODE)
    HASCAPITAL = re.compile(ur"^.*[[:upper:]].*$", re.UNICODE)
    LOWER_CASE = re.compile(ur"^([[:lower:]]+)+$", re.UNICODE)
    LOWER_PUNC = re.compile(ur"^([[:lower:]]+[[:punct:]]+)+$", re.UNICODE)
    OTHER_CASE = re.compile(ur"^.*$", re.UNICODE)  # Always matches

    @staticmethod
    def list():  # Fixes the iterator ordering bug in enum
        return [
            Feature.UPPER_CASE,
            Feature.CAMEL_CASE,
            Feature.UPPER_NAME,
            Feature.CAMEL_NAME,
            Feature.UPPER_PUNC,
            Feature.CAMEL_PUNC,
            Feature.FIRST_WORD,
            Feature.CAPITALIZE,
            Feature.HASCAPITAL,
            Feature.LOWER_CASE,
            Feature.LOWER_PUNC,
            Feature.OTHER_CASE
        ]

    @staticmethod
    def get(text):
        for feature in Feature.list():
            if feature.value.match(text):
                return feature
        return Feature.OTHER_CASE

    def index(self):
        return Feature.list().index(self)


class Model:
    def __init__(self):
        size = len(Marker.list())
        self.num_words = 0
        self.marker_selection_prob = [defaultdict(lambda: [0.0] * size) for _ in xrange(size)]
        self.first_selection_prob = [[defaultdict(float) for _ in xrange(size)] for _ in xrange(size)]
        self.next_selection_prob = [defaultdict(lambda: defaultdict(float)) for _ in xrange(size)]

        self.marker_backoff_prob_0 = [[0.0] * size for _ in xrange(size)]       # Without word_prev
        self.marker_backoff_prob_1 = [0.0] * size                               # Without marker_prev

        self.first_backoff_prob_0 = [defaultdict(float) for _ in xrange(size)]  # Without marker_prev
        self.first_backoff_prob_1 = [[defaultdict(float) for _ in xrange(size)] for _ in xrange(size)]

        self.next_backoff_prob_0 = [defaultdict(lambda: defaultdict(float)) for _ in xrange(size)]
        self.next_backoff_prob_1 = [defaultdict(float) for _ in xrange(size)]   # Without word_prev

    # Pr(NC | NC^-1, W^-1)
    def marker_prob(self, marker_prev, word_prev, marker_curr):
        prob = self.marker_selection_prob[marker_prev][word_prev][marker_curr]
        if not prob:
            prob = self.marker_backoff_prob_0[marker_prev][marker_curr] / self.num_words
        if not prob:
            prob = 1 / self.num_words / (len(Marker.list()) ** 2)
        return prob

    # Pr(WF_1 | NC, NC^-1)
    def first_prob(self, marker_prev, marker_curr, word_curr):
        prob = self.first_selection_prob[marker_prev][marker_curr][word_curr]
        if not prob:
            prob = self.first_backoff_prob_0[marker_curr][word_curr] / len(Marker.list())
        if not prob:
            prob = self.first_backoff_prob_1[marker_prev][marker_curr][Feature.get(word_curr)] / self.num_words * len(Feature.list())
        if not prob:
            prob = 1 / self.num_words / (len(Marker.list()) ** 2)
        return prob

    # Pr(WF | WF^-1, NC)
    def next_prob(self, marker_curr, word_prev, word_curr):
        prob = self.next_selection_prob[marker_curr][word_prev][word_curr]
        if not prob:
            prob = self.next_backoff_prob_0[marker_curr][word_prev][Feature.get(word_curr)] / self.num_words * len(Feature.list())
        if not prob:
            prob = 1 / (self.num_words ** 2) / len(Marker.list())
        return prob

    def fit(self, train_data):
        self.num_words += sum(map(len, train_data))

        print "Number of words: {}".format(self.num_words)

        size = len(Marker.list())
        word_start = Marker.START.value
        word_end = Marker.END.value
        marker_start = Marker.START.index()
        marker_end = Marker.END.index()

        # c(NC, NC^-1, W^-1) = marker_selections[NC^-1][W^-1][NC]
        marker_selections = [defaultdict(lambda: [0] * size) for _ in xrange(size)]

        # c(NC^-1, W^-1) = marker_counts[NC^-1][W^-1]
        marker_counts = [defaultdict(int) for _ in xrange(size)]

        # c(WF_first, NC, NC^-1) = marker_trans_wordf[NC^-1][NC][WF_first]
        marker_trans_wordf = [[defaultdict(int) for _ in xrange(size)] for _ in xrange(size)]

        # c(NC, NC^-1) = marker_trans[NC^-1][NC]
        marker_trans = [[0] * size] * size

        # c(WF, WF^-1, NC) = marker_wordf_trans[NC][WF^-1][WF]
        marker_wordf_trans = [defaultdict(lambda: defaultdict(int)) for _ in xrange(size)]

        # c(WF^-1, NC) = marker_wordf[NC][WF^-1]
        marker_wordf = [defaultdict(int) for _ in xrange(size)]

        for words_list in train_data:
            if not words_list:
                continue

            (word_prev, marker_prev) = (word_start, marker_start)  # Or word_start ?
            (word_curr, marker_curr) = words_list[0]

            marker_selections[marker_prev][word_prev][marker_curr] += 1
            marker_counts[marker_prev][word_prev] += 1
            marker_trans_wordf[marker_prev][marker_curr][word_curr] += 1
            marker_trans[marker_prev][marker_curr] += 1
            marker_wordf_trans[marker_curr][word_prev][word_curr] += 1
            marker_wordf[marker_curr][word_prev] += 1

            for ((word_prev, marker_prev), (word_curr, marker_curr)) in izip(words_list, words_list[1:]):
                if marker_prev != marker_curr:
                    marker_selections[marker_prev][word_prev][marker_curr] += 1
                    marker_counts[marker_prev][word_prev] += 1
                    marker_trans_wordf[marker_prev][marker_curr][word_curr] += 1
                    marker_trans[marker_prev][marker_curr] += 1
                    # Transition from start to curr
                    marker_wordf_trans[marker_curr][word_start][word_curr] += 1
                    marker_wordf[marker_curr][word_start] += 1
                    # Transition from prev to end
                    marker_wordf_trans[marker_prev][word_prev][word_end] += 1
                    marker_wordf[marker_prev][word_prev] += 1
                else:
                    marker_wordf_trans[marker_prev][word_prev][word_curr] += 1
                    marker_wordf[marker_curr][word_prev] += 1

            (word_prev, marker_prev) = words_list[-1]
            (word_curr, marker_curr) = (word_end, marker_end)

            marker_selections[marker_prev][word_prev][marker_curr] += 1
            marker_counts[marker_prev][word_prev] += 1
            marker_trans_wordf[marker_prev][marker_curr][word_curr] += 1
            marker_trans[marker_prev][marker_curr] += 1
            marker_wordf_trans[marker_prev][word_prev][word_curr] += 1
            marker_wordf[marker_prev][word_prev] += 1

        # marker_selection_prob
        for marker_prev in xrange(size):
            for (word_prev, denom) in marker_counts[marker_prev].iteritems():
                for marker_curr in xrange(size):
                    num = marker_selections[marker_prev][word_prev][marker_curr]
                    self.marker_selection_prob[marker_prev][word_prev][marker_curr] = num / denom

        # first_selection_prob
        for marker_prev in xrange(size):
            for marker_curr in xrange(size):
                denom = marker_trans[marker_prev][marker_curr]
                for (word_curr, num) in marker_trans_wordf[marker_prev][marker_curr].iteritems():
                    self.first_selection_prob[marker_prev][marker_curr][word_curr] = num / denom

        # next_selection_prob
        for marker_curr in xrange(size):
            for (word_prev, denom) in marker_wordf[marker_curr].iteritems():
                for (word_curr, num) in marker_wordf_trans[marker_curr][word_prev].iteritems():
                    self.next_selection_prob[marker_curr][word_prev][word_curr] = num / denom

        # marker_backoff_prob_0
        for marker_prev in xrange(size):
            marker_selections_0 = [0] * size
            marker_counts_0 = 0
            for (word_prev, denom) in marker_counts[marker_prev].iteritems():
                marker_counts_0 += denom
                for marker_curr in xrange(size):
                    num = marker_selections[marker_prev][word_prev][marker_curr]
                    marker_selections_0[marker_curr] += num
            for marker_curr in xrange(size):
                num = marker_selections_0[marker_curr]
                denom = marker_counts_0
                self.marker_backoff_prob_0[marker_prev][marker_curr] = num / denom if denom else 0.0

        # first_backoff_prob_0
        marker_trans_0 = [0] * size
        marker_trans_wordf_0 = [defaultdict(int) for _ in xrange(size)]
        for marker_prev in xrange(size):
            for marker_curr in xrange(size):
                denom = marker_trans[marker_prev][marker_curr]
                marker_trans_0[marker_curr] += denom
                for (word_curr, num) in marker_trans_wordf[marker_prev][marker_curr].iteritems():
                    marker_trans_wordf_0[marker_curr][word_curr] += num
                    self.first_selection_prob[marker_prev][marker_curr][word_curr] = num / denom
        for marker_curr in xrange(size):
            denom = marker_trans_0[marker_curr]
            for (word_curr, num) in marker_trans_wordf_0[marker_curr].iteritems():
                self.first_backoff_prob_0[marker_curr][word_curr] = num / denom

        # first_backoff_prob_1
        for marker_prev in xrange(size):
            for marker_curr in xrange(size):
                denom = marker_trans[marker_prev][marker_curr]
                marker_trans_wordf_0 = defaultdict(int)
                for (word_curr, num) in marker_trans_wordf[marker_prev][marker_curr].iteritems():
                    marker_trans_wordf_0[Feature.get(word_curr)] += num
                for (feature_curr, num) in marker_trans_wordf_0.iteritems():
                    self.first_backoff_prob_1[marker_prev][marker_curr][feature_curr] = num / denom

        # next_backoff_prob_0
        for marker_curr in xrange(size):
            for (word_prev, denom) in marker_wordf[marker_curr].iteritems():
                marker_wordf_trans_0 = defaultdict(int)
                for (word_curr, num) in marker_wordf_trans[marker_curr][word_prev].iteritems():
                    marker_wordf_trans_0[Feature.get(word_curr)] += num
                for (feature_curr, num) in marker_wordf_trans_0.iteritems():
                    self.next_backoff_prob_0[marker_curr][word_prev][feature_curr] = num / denom

    def extract(self, words_list):
        size = len(Marker.list())
        word_start = Marker.START.value
        word_end = Marker.END.value
        marker_start = Marker.START.index()
        marker_end = Marker.END.index()

        num_words = len(words_list)

        paths = [[-1] * size for _ in xrange(num_words)]
        probs_prev = [0.0] * size
        probs_prev[marker_start] = 1.0
        word_prev = word_start

        for step in xrange(num_words):
            # print "------ Step {} ------".format(step)
            (word_curr, _, _) = words_list[step]
            probs_curr = [0.0] * size
            for marker_prev in xrange(size):
                if marker_prev == marker_end or (step > 0 and marker_prev == marker_start):
                    continue
                for marker_curr in xrange(size):
                    if marker_curr == marker_start or marker_curr == marker_end:
                        continue
                    if marker_prev != marker_curr:
                        prob = self.marker_prob(marker_prev, word_prev, marker_curr)
                        prob *= self.first_prob(marker_prev, marker_curr, word_curr)
                    else:
                        prob = self.next_prob(marker_curr, word_prev, word_curr)
                        prob *= 1.0 - self.next_prob(marker_curr, word_prev, word_end)
                    # print "  {} -> {}: {}".format(Marker.get(marker_prev).name, Marker.get(marker_curr).name, prob)
                    prob *= probs_prev[marker_prev]
                    if prob > probs_curr[marker_curr]:
                        # print "probs_curr[{}] = {}".format(Marker.get(marker_curr).name, prob)
                        probs_curr[marker_curr] = prob
                        paths[step][marker_curr] = marker_prev
            # print probs_curr
            word_prev = word_curr
            probs_prev = probs_curr

        (word_curr, marker_curr) = (word_end, marker_end)
        prob_end = 0.0
        last_marker = 0
        for marker_prev in xrange(size):
            prob = self.next_prob(marker_curr, word_prev, word_end)
            prob *= probs_prev[marker_prev]
            if prob > prob_end:
                prob_end = prob
                last_marker = marker_prev

        result = []
        marker = last_marker
        for step in reversed(xrange(num_words)):
            result.append(marker)
            marker = paths[step][marker]

        result = zip(words_list, reversed(result))
        result = map(lambda ((_, start, end), marker_id): (Marker.get(marker_id).value, start, end), result)
        return result


hyphen_regexp = re.compile(ur"(?<=\w)[-‒–—―]\s+|\s+[-‒–—―](?=\w)", re.UNICODE)
word_regexp = re.compile(ur"\w+([^\w\s]+\w+)*", re.UNICODE)


def words(text, positions=False):
    return [(w.group(0), w.start(), w.end()) if positions else w.group(0) for w in re.finditer(word_regexp, text)]


def test_hmm(test_data):

    print "Preparing data..."

    train_data = []
    with gzip.open('train.db.ru.xml.gz') as train:
        document = XMLTree.parse(train).getroot()
        for sentence in document:
            words_list = zip(words(sentence.text), repeat(Marker.OTHER.index()))
            for collocation in sentence:
                collocation_text = re.sub(hyphen_regexp, "-", collocation.text)  # Hello from line 13403
                collocation_marker = Marker.OTHER
                for (key, value) in collocation.items():
                    (key, value) = (key.upper(), value.upper())  # Hello from line 2400
                    if key == "TYPE":
                        collocation_marker = Marker[value]
                words_list.extend(zip(words(collocation_text), repeat(collocation_marker.index())))
                words_list.extend(zip(words(collocation.tail), repeat(Marker.OTHER.index())))
            train_data.append(words_list)
            # print " ".join(map(lambda (word, marker_index): Marker.get(marker_index).colorize(word), words_list))

    print "Training..."

    model = Model()
    model.fit(train_data)

    print "Testing..."

    result = []
    for sentence in test_data:
        words_list = words(sentence, positions=True)
        collocations = model.extract(words_list)
        result.append(collocations)
        print_sentence(sentence, collocations)
    return result


def print_sentence(sentence, collocations):
    index = 0
    for (value, start, end) in collocations:
        marker = Marker(value)
        sys.stdout.write(sentence[index:start])
        sys.stdout.write(marker.colorize(sentence[start:end]))
        index = end
    sys.stdout.write(sentence[index:] if collocations else sentence)
    sys.stdout.write("\n")


if __name__ == '__main__':
    with open("test.txt", "r") as test:
        sentences_data = map(lambda line: line.decode('utf-8').strip(), test.readlines())
        collocations_data = test_hmm(sentences_data)
        # for (s, c) in zip(sentences_data, collocations_data):
        #     print_sentence(s, c)
