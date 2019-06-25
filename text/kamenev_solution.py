# -*- coding: utf-8 -*-
import string
from xml.etree import ElementTree
from collections import defaultdict
from math import log
from copy import deepcopy

TAGS_SHORT = ['P', 'C', 'G']
TAGS_LONG = ['PERSON', 'ORGANIZATION', 'LOCATION', 'OTHER']
INF_MINUS = -1e8


class Markov:
    def __init__(self):
        self.transitition_prob = []
        self.result_prob = []
        self.transititions = []
        self.tag_starter = []
        self.nodes = []
        self.word_counter = 0

    def fit(self, input):
        global TAGS_LONG
        self.transitition_prob = [([0.] * len(TAGS_LONG))] * len(TAGS_LONG)
        self.tag_starter = [0.] * len(TAGS_LONG)
        self.nodes = list(TAGS_LONG)
        self.result_prob = [[{} for i in xrange(len(TAGS_LONG))] for j in xrange(len(TAGS_LONG))]
        tag_counter = [0] * len(TAGS_LONG)
        raw_trans = [[defaultdict(int) for i in xrange(len(TAGS_LONG))] for j in xrange(len(TAGS_LONG))]
        trans_cnt = [[0] * len(TAGS_LONG)] * len(TAGS_LONG)
        starting_tag_counter = [0] * len(TAGS_LONG)

        for x in input:
            self.word_counter += len(x)

            w, tag = x[0]
            starting_tag_counter[tag] += 1
            tag_counter[tag] += 1

            for j in xrange(1, len(x)):
                w_i, tag_i = x[j - 1]
                w_j, tag_j = x[j]

                tag_counter[tag_j] += 1
                trans_cnt[tag_i][tag_j] += 1
                raw_trans[tag_i][tag_j][w_j] += 1

        self.transititions = deepcopy(trans_cnt)
        for i in xrange(len(TAGS_LONG)):
            self.tag_starter[i] = log(starting_tag_counter[i]) - log(len(input))

            for j in xrange(len(TAGS_LONG)):

                self.transitition_prob[i][j] = log(trans_cnt[i][j]) - log(tag_counter[i])

                for w, cnt in raw_trans[i][j].items():
                    self.result_prob[i][j][w] = log(cnt) - log(trans_cnt[i][j])


def my_filter(text):
    return filter(lambda x: x not in string.punctuation, text)


def extract_sentences(train_input, valid_tags):
    tree = ElementTree.parse(train_input).getroot()

    res = []
    for sentence in tree:
        words = my_filter(sentence.text).split()
        tags = [valid_tags.index('OTHER')] * len(words)

        for elem in sentence:
            tag = valid_tags.index(elem.get(elem.keys()[0]))
            words.append(my_filter(elem.text))
            tags.append(tag)

            tail_words = my_filter(elem.tail).split()
            words += tail_words
            tags += [valid_tags.index('OTHER')] * len(tail_words)

        res.append(zip(words, tags))

    return res


def apply_model(words, model):
    word_count = len(words)
    tag_count = 4

    tagged_entity_max_word_count = 3
    probab = [[INF_MINUS] * tag_count for _ in xrange(word_count)]
    path = [[(-1, -1)] * tag_count for _ in xrange(word_count)]
    probab[0] = deepcopy(model.tag_starter)

    for t in xrange(1, word_count):
        for j in xrange(tag_count):
            entitiy_word_start = tagged_entity_max_word_count

            if t < entitiy_word_start:
                entitiy_word_start = t

            if model.nodes[j] == 'OTHER':
                entitiy_word_start = 1

            for k in xrange(entitiy_word_start):
                o = ' '.join(words[t - k:t + 1])

                for i in xrange(tag_count):
                    if o in model.result_prob[i][j]:
                        result = model.result_prob[i][j][o]
                    else:
                        result = -log(model.word_counter) * len(o.split())

                    p = probab[t - k - 1][i] + model.transitition_prob[i][j] + result

                    if probab[t][j] < p:
                        probab[t][j] = p
                        path[t][j] = (t - k - 1, i)

    tag_end = word_count - 1
    tag = 0

    for j in xrange(tag_count):
        if probab[tag_end][tag] < probab[word_count - 1][j]:
            tag = j

    res = []
    while tag_end >= 0:
        prev_tag_end, prev_tag = path[tag_end][tag]
        res.append((tag, prev_tag_end + 1, tag_end))
        tag_end = prev_tag_end
        tag = prev_tag

    return res


def test_hmm(test_sentences):
    global TAGS_LONG
    train_input = open('train.db.ru.xml', 'r')
    train_sentences = extract_sentences(train_input, TAGS_LONG)

    model = Markov()
    model.fit(train_sentences)

    answer = []
    for sentence in test_sentences:
        test_words = my_filter(sentence).split()
        res = apply_model(test_words, model)
        answer.append(map(lambda ans: (TAGS_SHORT[ans[0]], ans[1], ans[2]), reversed(filter(lambda r: r[0] < 3, res))))

    return answer

test_string = u'российская легкоатлетка Елена Исинбаева установила новый мировой рекорд в прыжках с шестом взяв высоту 5 04 метра на турнире супер гран при в Монако'
test2 = u"25 апреля во Владивостоке состоялись первые в истории переговоры Владимира Путина с Ким Чен Ыном, который с 2011 года возглавляет КНДР."
print test_hmm([test2])
