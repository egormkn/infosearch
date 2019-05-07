#!/usr/bin/env python2
# coding: utf-8

from __future__ import division

import sys
import argparse
import unicodecsv as csv
import heapq
import math
import re
import cPickle as pickle
from abc import abstractmethod, ABCMeta
from operator import itemgetter as get, mul
from itertools import izip, tee, chain
from collections import defaultdict
from transliterate import get_translit_function


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)


def defaultdict_factory():
    return defaultdict(float)


class Phonetic(object):
    __metaclass__ = ABCMeta

    lang_ru = re.compile(ur"^[А-Я]+$", re.I | re.U)
    lang_en = re.compile(ur"^[A-Z]+$", re.I | re.U)

    def __init__(self):
        self.data = defaultdict(list)

    def get_results(self, word, limit=100):
        results = []
        if re.match(self.lang_ru, word):
            results.extend(self.data[self.hash_ru(word)])
        if re.match(self.lang_en, word):
            results.extend(self.data[self.hash_en(word)])
        results = map(get(1), sorted(results, key=get(0), reverse=True))
        if limit > 0:
            results = results[:limit]
        return results

    def add(self, word, prob):
        if re.match(self.lang_ru, word):
            self.data[self.hash_ru(word)].append((prob, word))
        if re.match(self.lang_en, word):
            self.data[self.hash_en(word)].append((prob, word))

    @abstractmethod
    def hash_ru(self, word):
        pass

    @abstractmethod
    def hash_en(self, word):
        pass


class Metaphone(Phonetic):
    replaces_en = map(lambda (pattern, replace): (re.compile(pattern, re.I | re.U), replace), [
        (ur"([^C])+(?=\1)", ur""),  # Drop duplicate adjacent letters, except for C
        (ur"^[KGP](?=N)", ur""),
        (ur"^A(?=E)", ur""),
        (ur"^W(?=R)", ur""),
        (ur"(?<=M)B$", ur""),
        (ur"SCH", ur"SKH"),
        (ur"C(?=IA|H)", ur"X"),
        (ur"C(?=[IEY])", ur"S"),
        (ur"C", ur"K"),
        (ur"D(?=G[EIY])", ur"J"),
        (ur"D", ur"T"),
        (ur"G(?=H[^AEIOUY])", ur""),
        (ur"G(?=N(ED)?$)", ur""),
        (ur"(?<!G)G(?=[IEY])", ur"J"),
        (ur"G", ur"K"),
        (ur"(?<=[AEIOUY])H(?=[^AEIOUY])", ur""),
        (ur"CK", ur"K"),
        (ur"PH", ur"F"),
        (ur"Q", ur"K"),
        (ur"S(?=H|IO|IA)", ur"X"),
        (ur"T(?=IA|IO)", ur"X"),
        (ur"TH", ur"0"),
        (ur"T(?=CH)", ur""),
        (ur"V", ur"F"),
        (ur"^WH", ur"W"),
        (ur"W(?=[^AEIOUY]|$)", ur""),
        (ur"^X", ur"S"),
        (ur"X", ur"KS"),
        (ur"Y(?=[^AEIOUY]|$)", ur""),
        (ur"Z", ur"S"),
        (ur"(?<=.)[AEIOUY]", ur"")
    ])
    replaces_ru = map(lambda (pattern, replace): (re.compile(pattern, re.I | re.U), replace), [
        (ur"[ИЙ][ОЕ]", ur"И"),
        (ur"[ОЫЯ]", ur"А"),
        (ur"[ЕЁЭ]", ur"И"),
        (ur"Ю", ur"У"),
        (ur"Б(?![АИУЛМНР])", ur"П"),
        (ur"З(?![АИУЛМНР])", ur"С"),
        (ur"Д(?![АИУЛМНР])", ur"Т"),
        (ur"В(?![АИУЛМНР])", ur"Ф"),
        (ur"Г(?![АИУЛМНР])", ur"К"),
        (ur"[ТД]С", ur"Ц")
    ])

    def hash_ru(self, word):
        code = word
        for (pattern, replace) in Metaphone.replaces_ru:
            code = re.sub(pattern, replace, code)
        return code

    def hash_en(self, word):
        code = word
        for (pattern, replace) in Metaphone.replaces_en:
            code = re.sub(pattern, replace, code)
        return code


class Trie(object):
    nodes = []

    def __init__(self, char, start=False):
        self.char = char
        self.query = None
        self.max_prob = 0.0
        self.children = {}
        if start:
            self.id = -1
        else:
            self.id = len(Trie.nodes)
            Trie.nodes.append(self)

    def get_children(self):
        return {c: Trie.nodes[i] for c, i in self.children.iteritems()}

    def get_child(self, c):
        return Trie.nodes[self.children[c]]

    def set_child(self, c, child):
        self.children[c] = child.id

    def add(self, word, prob=0.0):
        trie = self
        trie.max_prob = max(trie.max_prob, prob)
        for c in word + "$":
            if c not in trie.children:
                child = Trie(c)
                trie.set_child(c, child)
            else:
                child = trie.get_child(c)
            child.max_prob = max(child.max_prob, prob)
            trie = child
        trie.query = word

    def is_end(self):
        return self.char == "$"

    def get_results(self, word, edit_prob, default_prob, best_score, limit=100):
        word = "^" + word + "$"
        result = []
        queue = []
        heapq.heappush(queue, (-1.0, 0, self, [('M', '^', '^')]))
        while len(queue) > 0:
            path_prev = heapq.heappop(queue)
            (prob_prev, pos, node_prev, history_prev) = path_prev  # type: (float, int, Trie, list)
            if pos + 1 < len(word):
                # --- Process replace/match ---
                c_correct_prev = node_prev.char
                c_error_prev = word[pos]
                c_error_curr = word[pos + 1]
                for c_correct_curr in node_prev.children:
                    node_curr = node_prev.get_child(c_correct_curr)
                    if c_correct_curr == c_error_curr:
                        prob_curr = 1.0
                        history_curr = history_prev + [('M', c_correct_curr, c_error_curr)]
                    else:
                        prob_curr = edit_prob[c_correct_curr][c_error_curr]
                        if not prob_curr:
                            prob_curr = default_prob
                        history_curr = history_prev + [('R', c_correct_curr, c_error_curr)]
                    prob_curr *= prob_prev
                    path_curr = (prob_curr, pos + 1, node_curr, history_curr)
                    if -prob_curr * node_curr.max_prob ** SpellChecker.lambda_coeff > best_score:
                        heapq.heappush(queue, path_curr)
                # --- Process addition ---
                c_correct_prev = node_prev.char
                c_error_prev = word[pos]
                c_correct_curr = ' '
                c_error_curr = word[pos + 1]
                node_curr = node_prev
                prob_curr = edit_prob[c_correct_prev + c_correct_curr][c_correct_prev + c_error_curr]
                if not prob_curr:
                    prob_curr = default_prob
                prob_curr *= prob_prev
                history_curr = history_prev + [('I', ' ', c_error_curr)]
                path_curr = (prob_curr, pos + 1, node_curr, history_curr)
                if -prob_curr * node_curr.max_prob ** SpellChecker.lambda_coeff > best_score:
                    heapq.heappush(queue, path_curr)
                # --- Process deletion ---
                c_correct_prev = node_prev.char
                c_error_prev = word[pos]
                c_error_curr = ' '
                for c_correct_curr in node_prev.children:
                    node_curr = node_prev.get_child(c_correct_curr)
                    prob_curr = edit_prob[c_correct_prev + c_correct_curr][c_correct_prev + c_error_curr]
                    if not prob_curr:
                        prob_curr = default_prob
                    history_curr = history_prev + [('D', c_correct_curr, ' ')]
                    prob_curr *= prob_prev
                    path_curr = (prob_curr, pos, node_curr, history_curr)
                    if -prob_curr * node_curr.max_prob ** SpellChecker.lambda_coeff > best_score:
                        heapq.heappush(queue, path_curr)
                # --- Process transpose ---
                if pos + 2 < len(word):
                    c_correct_prev = node_prev.char
                    c_error_prev = word[pos]
                    c_error_curr1 = word[pos + 1]
                    c_error_curr2 = word[pos + 2]
                    c_correct_curr1 = c_error_curr2
                    c_correct_curr2 = c_error_curr1
                    if c_correct_curr1 in node_prev.children:
                        node_curr1 = node_prev.get_child(c_correct_curr1)
                        if c_correct_curr2 in node_curr1.children:
                            node_curr2 = node_curr1.get_child(c_correct_curr2)
                            prob_curr = edit_prob[c_correct_curr1 + c_correct_curr2][c_error_curr1 + c_error_curr2]
                            if not prob_curr:
                                prob_curr = default_prob
                            history_curr = history_prev + [
                                ('T', c_correct_curr1, c_error_curr1),
                                ('T', c_correct_curr2, c_error_curr2)
                            ]
                            prob_curr *= prob_prev
                            path_curr = (prob_curr, pos + 2, node_curr2, history_curr)
                            if -prob_curr * node_curr.max_prob ** SpellChecker.lambda_coeff > best_score:
                                heapq.heappush(queue, path_curr)
            elif node_prev.is_end():
                result.append(node_prev.query)
                if len(result) >= limit > 0:
                    queue = []
                    return result
            else:
                c_correct_prev = node_prev.char
                c_error_curr = ' '
                for c_correct_curr in node_prev.children:
                    node_curr = node_prev.get_child(c_correct_curr)
                    prob_curr = edit_prob[c_correct_prev + c_correct_curr][c_correct_prev + c_error_curr]
                    if not prob_curr:
                        prob_curr = default_prob
                    history_curr = history_prev + [('D', c_correct_curr, ' ')]
                    prob_curr *= prob_prev
                    path_curr = (prob_curr, pos, node_curr, history_curr)
                    if -prob_curr * node_curr.max_prob ** SpellChecker.lambda_coeff > best_score:
                        heapq.heappush(queue, path_curr)
        queue = []
        return result


class SpellChecker(object):
    layout_dict = {
        ("EN", "RU"): {
            u"\"": u"Э", u"~": u"Ё", u"`": u"ё", u"q": u"й", u"w": u"ц", u"e": u"у", u"r": u"к", u"t": u"е", u"y": u"н",
            u"u": u"г",
            u"i": u"ш", u"o": u"щ", u"p": u"з", u"[": u"х", u"]": u"ъ", u"a": u"ф", u"s": u"ы", u"d": u"в", u"f": u"а",
            u"g": u"п",
            u"h": u"р", u"j": u"о", u"k": u"л", u"l": u"д", u";": u"ж", u"'": u"э", u"z": u"я", u"x": u"ч", u"c": u"с",
            u"v": u"м",
            u"b": u"и", u"n": u"т", u"m": u"ь", u",": u"б", u".": u"ю", u"/": u".", u"Q": u"Й", u"W": u"Ц", u"E": u"У",
            u"R": u"К",
            u"T": u"Е", u"Y": u"Н", u"U": u"Г", u"I": u"Ш", u"O": u"Щ", u"P": u"З", u"{": u"Х", u"}": u"Ъ", u"A": u"Ф",
            u"S": u"Ы",
            u"D": u"В", u"F": u"А", u"G": u"П", u"H": u"Р", u"J": u"О", u"K": u"Л", u"L": u"Д", u":": u"Ж", u"Z": u"Я",
            u"X": u"Ч",
            u"C": u"С", u"V": u"М", u"B": u"И", u"N": u"Т", u"M": u"Ь", u"<": u"Б", u">": u"Ю", u"?": u",", u"@": u"\"",
            u"#": u"№",
            u"$": u";", u"^": u":", u"&": u"?"
        },
        ("RU", "EN"): {
            u"Ё": u"~", u"ё": u"`", u"й": u"q", u"ц": u"w", u"у": u"e", u"к": u"r", u"е": u"t", u"н": u"y", u"г": u"u",
            u"ш": u"i",
            u"щ": u"o", u"з": u"p", u"х": u"[", u"ъ": u"]", u"ф": u"a", u"ы": u"s", u"в": u"d", u"а": u"f", u"п": u"g",
            u"р": u"h",
            u"о": u"j", u"л": u"k", u"д": u"l", u"э": u"'", u"я": u"z", u"ч": u"x", u"с": u"c", u"м": u"v", u"и": u"b",
            u"т": u"n",
            u"ь": u"m", u"Й": u"Q", u"Ц": u"W", u"У": u"E", u"К": u"R", u"Е": u"T", u"Н": u"Y", u"Г": u"U", u"Ш": u"I",
            u"Щ": u"O",
            u"З": u"P", u"Х": u"{", u"Ъ": u"}", u"Ф": u"A", u"Ы": u"S", u"В": u"D", u"А": u"F", u"П": u"G", u"Р": u"H",
            u"О": u"J",
            u"Л": u"K", u"Д": u"L", u"Э": u"\"", u"Я": u"Z", u"Ч": u"X", u"С": u"C", u"М": u"V", u"И": u"B", u"Т": u"N",
            u"Ь": u"M",
            u"Б": u"<", u"Ю": u">", u"\"": u"@", u"№": u"#", u";": u"$", u":": u"^", u"?": u"&", u",": u"?", u"Ж": u":",
            u"б": u",",
            u".": u"/", u"ю": u".", u"ж": u";"
        }
    }
    similar_re = {
        "EN": re.compile(ur"^[ABCEHKMOPTXY]+$", re.I | re.U),
        "RU": re.compile(ur"^[АВСЕНКМОРТХУ]+$", re.I | re.U)
    }
    multiplier_re = re.compile(ur"(?<=\d)Х(?=\d)|^Х(?=\d)|(?<=\d)Х$", re.I | re.U)
    layout_re = {
        "RU": re.compile(ur"^[^a-zA-Z]*[а-яА-Я][^a-zA-Z]*$", re.I | re.U),
        "EN": re.compile(ur"^[^а-яА-Я]*[a-zA-Z][^а-яА-Я]*$", re.I | re.U)
    }
    translit_fn = {
        "RU": get_translit_function("ru")
    }
    lambda_coeff = 0.5

    def __init__(self):
        self.num_queries = 0  # Total number of queries
        self.num_known_queries = 0  # Number of queries for which we know the correction
        self.known_answers = {}  # Dictionary of training data
        self.word_prob = defaultdict(float)  # Language model
        self.edit_prob = defaultdict(float)  # Zero-level error model
        self.unigram_edit_prob = defaultdict(defaultdict_factory)  # First-level error model
        self.bigram_edit_prob = defaultdict(defaultdict_factory)  # Second-level error model
        self.smart_edit_prob = defaultdict(defaultdict_factory)  # Kernighan-Church-Gale error model
        self.layout_prob = {("RU", "EN"): 0.0, ("EN", "RU"): 0.0}  # Layout error model
        self.translit_prob = {("RU", "EN"): 0.0, ("EN", "RU"): 0.0}  # Translit error model
        self.metaphone = Metaphone()  # Metaphone word lists
        self.trie = Trie("^", start=True)  # Dictionary trie

    def fit(self, corrections, frequency):

        # Build language model
        frequency_size = len(frequency)
        for (i, (word, num_queries)) in enumerate(frequency):
            if not i % 10000:
                print "Building language model:", i, "/", frequency_size
            self.num_queries += num_queries
            self.word_prob[word] += num_queries
        for word in self.word_prob:
            self.word_prob[word] /= self.num_queries

        num_unigram_edits = 0
        num_bigram_edits = 0
        num_smart_edits = defaultdict(int)

        # Build error model (3 levels)
        corrections_size = len(corrections)
        for (i, (word, correction, num_queries)) in enumerate(corrections):
            if not i % 10000:
                print "Building error model:", i, "/", corrections_size
            self.num_known_queries += num_queries
            self.known_answers[word] = correction

            is_lang_error = False

            for lang_pair in self.layout_prob:  # lang_pair = (lang_desired, lang_used)
                correction_rev = self.change_layout(correction, lang_pair)
                if correction != word and correction_rev == word:
                    self.layout_prob[lang_pair] += num_queries
                    is_lang_error = True

            for lang_pair in self.translit_prob:  # lang_pair = (lang_desired, lang_used)
                correction_rev = SpellChecker.transliterate(correction, lang_pair)
                if correction != word and correction_rev == word:
                    self.translit_prob[lang_pair] += num_queries
                    is_lang_error = True

            if is_lang_error or re.search(SpellChecker.multiplier_re, word):
                continue
            elif re.match(SpellChecker.layout_re["RU"], correction) and re.match(SpellChecker.layout_re["EN"], word):
                print "Not processed:", word, "~>", correction
            elif re.match(SpellChecker.layout_re["EN"], correction) and re.match(SpellChecker.layout_re["RU"], word):
                print "Not processed:", word, "~>", correction

            (distance, prescription) = self.distance(correction, word)
            for (edit, c_correct, c_error) in prescription:
                num_unigram_edits += num_queries
                self.edit_prob[edit] += num_queries
                self.unigram_edit_prob[c_correct][c_error] += num_queries

            prescription = [('M', '^', '^')] + prescription + [('M', '$', '$')]  # Add start and end symbols
            pairs = pairwise(prescription)
            prev_char = "^"
            for ((edit_prev, c_correct_prev, c_error_prev), (edit_curr, c_correct_curr, c_error_curr)) in pairs:
                c_correct = c_correct_prev + c_correct_curr
                c_error = c_error_prev + c_error_curr
                num_bigram_edits += num_queries
                self.bigram_edit_prob[c_correct][c_error] += num_queries
                if edit_curr == 'T' and c_correct_prev == c_error_curr and c_error_prev == c_correct_curr:
                    self.smart_edit_prob[c_correct][c_error] += num_queries
                elif edit_curr == 'D' or edit_curr == 'I':
                    self.smart_edit_prob[prev_char + c_correct_curr][prev_char + c_error_curr] += num_queries
                elif edit_curr == 'R':
                    self.smart_edit_prob[c_correct_curr][c_error_curr] += num_queries
                if c_correct_curr != " ":
                    prev_char = c_correct_curr

            formatted = "^" + correction + "$"
            for char in formatted:
                num_smart_edits[char] += num_queries
            for (char_prev, char_curr) in pairwise(formatted):
                num_smart_edits[char_prev + char_curr] += num_queries

        for lang_pair in self.layout_prob:
            self.layout_prob[lang_pair] /= self.num_known_queries

        for lang_pair in self.translit_prob:
            self.translit_prob[lang_pair] /= self.num_known_queries

        for edit in self.edit_prob:
            self.edit_prob[edit] /= num_unigram_edits

        for c_correct in self.unigram_edit_prob:
            for c_error in self.unigram_edit_prob[c_correct]:
                self.unigram_edit_prob[c_correct][c_error] /= num_unigram_edits

        for c_correct in self.bigram_edit_prob:
            for c_error in self.bigram_edit_prob[c_correct]:
                self.bigram_edit_prob[c_correct][c_error] /= num_bigram_edits

        for c_correct in self.smart_edit_prob:
            for c_error in self.smart_edit_prob[c_correct]:
                if len(c_correct) == 2 and c_correct[1] == ' ':  # Insert
                    self.smart_edit_prob[c_correct][c_error] /= num_smart_edits[c_correct[0]]
                else:  # Replace, delete or transpose
                    self.smart_edit_prob[c_correct][c_error] /= num_smart_edits[c_correct]

        # Build trie and metaphone
        for (i, (word, num_queries)) in enumerate(frequency):
            if not i % 10000:
                print "Building trie + metaphone:", i, "/", len(frequency)

            prob = self.word_prob[word]
            self.trie.add(word, prob)
            self.metaphone.add(word, prob)

        Trie.edit_prob = self.smart_edit_prob

        for lang_pair in self.layout_prob:
            print "Layout change probability:", lang_pair, "->", self.layout_prob[lang_pair]

        for lang_pair in self.translit_prob:
            print "Translit probability:", lang_pair, "->", self.translit_prob[lang_pair]

        for edit in self.edit_prob:
            print "Edit probability:", edit, "->", self.edit_prob[edit]

    def error_prob(self, query, intent):  # P(q|c)
        result_prob = 1.0

        (distance, prescription) = self.distance(intent, query)
        prescription = [('M', '^', '^')] + prescription + [('M', '$', '$')]  # Add start and end symbols
        pairs = pairwise(prescription)
        prev_char = "^"
        for ((edit_prev, c_correct_prev, c_error_prev), (edit_curr, c_correct_curr, c_error_curr)) in pairs:
            c_correct = c_correct_prev + c_correct_curr
            c_error = c_error_prev + c_error_curr
            if edit_curr == 'T' and c_correct_prev == c_error_curr and c_error_prev == c_correct_curr:
                prob = self.smart_edit_prob[c_correct][c_error]
                result_prob *= prob if prob else 1 / self.num_queries
            elif edit_curr == 'D' or edit_curr == 'I':
                prob = self.smart_edit_prob[prev_char + c_correct_curr][prev_char + c_error_curr]
                result_prob *= prob if prob else 1 / self.num_queries
            elif edit_curr == 'R':
                prob = self.smart_edit_prob[c_correct_curr][c_error_curr]
                result_prob *= prob if prob else 1 / self.num_queries
            if c_correct_curr != " ":
                prev_char = c_correct_curr

        return result_prob

        # model1_distance = self.distance(intent, query,
        #                                 delete_cost=lambda c: math.log(1.0 - self.unigram_edit_prob[c][' ']),
        #                                 insert_cost=lambda c: math.log(1.0 - self.unigram_edit_prob[' '][c]),
        #                                 replace_cost=lambda c1, c2: math.log(1.0 - self.unigram_edit_prob[c1][c2]),
        #                                 transpose_cost=lambda c1, c2: math.log(1.0 - self.bigram_edit_prob[c1+c2][c2+c1]))
        # (distance, prescription) = model1_distance
        # model_probs[1] = math.e ** distance
        #
        # print model_probs
        # return sum(map(mul, model_probs, model_coeffs))
        # weighted_distance = self.distance(correction, word,
        #                                   delete_cost=lambda c: 1.0 - self.unigram_edit_prob[' '][c],
        #                                   insert_cost=lambda c: 1.0 - self.unigram_edit_prob[c][' '],
        #                                   replace_cost=lambda c1, c2: 1.0 - self.unigram_edit_prob[c1][c2],
        #                                   transpose_cost=lambda c1, c2: 1.0 - self.bigram_edit_prob[c1 + c2][c2 + c1])
        # (distance, prescription) = weighted_distance
        # prescription = [('M', '^', '^')] + prescription + [('M', '$', '$')]  # Add start and end symbols
        # pairs = zip(prescription, prescription[1:])
        # for ((edit_prev, c_correct_prev, c_error_prev), (edit_curr, c_correct_curr, c_error_curr)) in pairs:
        #     c_correct = c_correct_prev + c_correct_curr
        #     c_error = c_error_prev + c_error_curr
        #     prob = self.bigram_edit_prob[c_correct][c_error]
        #     result *= prob if prob else 1 / self.num_queries
        # return result
        # return 1 / (math.e ** distance)

    def intent_prob(self, correction):  # P(c)
        return self.word_prob[correction]

    def fix(self, word):
        if word in self.known_answers:
            best_word = self.known_answers[word]
            print word, "->", best_word, "(known)"
            return best_word
        initial_intent_prob = self.intent_prob(word)
        initial_error_prob = self.error_prob(word, word) ** SpellChecker.lambda_coeff
        best_word = word
        best_score = initial_intent_prob * initial_error_prob
        metaphone_results = self.metaphone.get_results(word, limit=100)
        trie_results = self.trie.get_results(word, self.smart_edit_prob, 1 / self.num_queries,
                                             best_score=initial_intent_prob * initial_error_prob, limit=100)
        for candidate in chain(metaphone_results, trie_results):
            intent_prob = self.intent_prob(candidate)
            error_prob = self.error_prob(word, candidate) ** SpellChecker.lambda_coeff
            score = intent_prob * error_prob
            if score > best_score:
                best_score, best_word = score, candidate
        for (lang_desired, lang_used) in self.layout_prob:
            candidate = self.change_layout(word, (lang_used, lang_desired))
            intent_prob = self.intent_prob(candidate)
            error_prob = self.layout_prob[(lang_desired, lang_used)] ** SpellChecker.lambda_coeff
            score = intent_prob * error_prob
            if score > best_score:
                best_score, best_word = score, candidate
        if best_word != word:
            print word, "->", best_word  # , "\t\t", repr(metaphone_results).decode('unicode-escape')
        elif re.search(SpellChecker.multiplier_re, word):
            best_word = re.sub(SpellChecker.multiplier_re, "X", word)
            print word, "->", best_word
            return best_word
        return best_word

    @staticmethod
    def distance(word1, word2,
                 delete_cost=lambda c: 1.0,
                 insert_cost=lambda c: 1.0,
                 replace_cost=lambda c1, c2: 0.0 if c1 == c2 else 1.0,
                 transpose_cost=lambda c1, c2: 1.0,
                 epsilon=' '):
        """Optimal string alignment distance (assumes that no substring is edited more than once)"""
        inf = float('inf')
        d = [[inf for _ in xrange(len(word2) + 1)] for _ in xrange(len(word1) + 1)]
        p = [['?' for _ in xrange(len(word2) + 1)] for _ in xrange(len(word1) + 1)]
        d[0][0] = 0.0
        p[0][0] = 'M'
        for i in xrange(1, len(word1) + 1):
            d[i][0] = d[i - 1][0] + delete_cost(word1[i - 1])
            p[i][0] = 'D'
        for j in xrange(1, len(word2) + 1):
            d[0][j] = d[0][j - 1] + insert_cost(word2[j - 1])
            p[0][j] = 'I'
        for i in xrange(1, len(word1) + 1):
            for j in xrange(1, len(word2) + 1):
                d[i][j] = d[i - 1][j - 1] + replace_cost(word1[i - 1], word2[j - 1])
                p[i][j] = 'M' if word1[i - 1] == word2[j - 1] else 'R'
                d_delete = d[i - 1][j] + delete_cost(word1[i - 1])
                if d_delete < d[i][j]:
                    d[i][j] = d_delete
                    p[i][j] = 'D'
                d_insert = d[i][j - 1] + insert_cost(word2[j - 1])
                if d_insert < d[i][j]:
                    d[i][j] = d_insert
                    p[i][j] = 'I'
                if i > 1 and j > 1 and word1[i - 1] == word2[j - 2] and word1[i - 2] == word2[j - 1]:
                    d_transpose = d[i - 2][j - 2] + transpose_cost(word1[i - 2], word1[i - 1])
                    if d_transpose < d[i][j]:
                        d[i][j] = d_transpose
                        p[i][j] = 'T'
        distance = d[len(word1)][len(word2)]

        prescription = []
        (i, j) = (len(word1), len(word2))
        while i > 0 or j > 0:
            if p[i][j] == 'M' or p[i][j] == 'R':
                prescription.append((p[i][j], word1[i - 1], word2[j - 1]))
                (i, j) = (i - 1, j - 1)
            elif p[i][j] == 'D':
                prescription.append((p[i][j], word1[i - 1], epsilon))
                (i, j) = (i - 1, j)
            elif p[i][j] == 'I':
                prescription.append((p[i][j], epsilon, word2[j - 1]))
                (i, j) = (i, j - 1)
            elif p[i][j] == 'T':
                prescription.append((p[i][j], word1[i - 1], word2[j - 1]))
                prescription.append((p[i][j], word1[i - 2], word2[j - 2]))
                (i, j) = (i - 2, j - 2)
        prescription.reverse()

        # print " ", list(" " + word2)
        # for (letter1, row) in zip(" " + word1, p):
        #     print letter1, row
        # print "".join(map(lambda (op, c1, c2): op, prescription))
        # print "".join(map(lambda (op, c1, c2): c1, prescription))
        # print "".join(map(lambda (op, c1, c2): c2, prescription))
        # print "Distance:", distance

        return distance, prescription

    @staticmethod
    def change_layout(word, lang=("RU", "EN")):
        (lang_from, lang_to) = lang
        if not re.match(SpellChecker.layout_re[lang_from], word):
            return word
        lang_dict = SpellChecker.layout_dict[lang]
        return "".join(lang_dict.get(c, c) for c in word)

    @staticmethod
    def transliterate(word, lang=("RU", "EN")):
        lang_from, lang_to = lang
        if lang_to == "EN":
            return SpellChecker.translit_fn[lang_from](word, reversed=True)
        else:
            return SpellChecker.translit_fn[lang_to](word, reversed=False)


def main():
    parser = argparse.ArgumentParser(description="Spellchecker")
    parser.add_argument("--cache", help="Use cached language and error models", action="store_true")
    args = parser.parse_args()

    if args.cache:
        print "Loading model from file:", "spellchecker.bin"

        with open("spellchecker.bin", "r") as file_dump:
            model = pickle.load(file_dump)
            Trie.nodes = pickle.load(file_dump)
        file_dump.close()

        print "Model loaded from spellchecker.bin"
    else:
        with open("public.freq.csv", "r") as correction_file:
            reader = csv.reader(correction_file, encoding='utf-8')
            next(reader, None)  # skip the header
            correction_data = map(lambda (orig, corr, freq): (orig, corr, int(freq)), reader)

        with open("words.freq.csv", "r") as frequency_file:
            reader = csv.reader(frequency_file, encoding='utf-8')
            next(reader, None)  # skip the header
            frequency_data = map(lambda (orig, freq): (orig, int(freq)), reader)

        model = SpellChecker()
        model.fit(correction_data, frequency_data)

        print "Saving model to file:", "spellchecker.bin"

        with open("spellchecker.bin", "w") as file_dump:
            pickle.dump(model, file_dump)
            pickle.dump(Trie.nodes, file_dump)
        file_dump.close()

        print "Model saved to spellchecker.bin"

    with open("no_fix.submission.csv", "r") as test_file:
        reader = csv.reader(test_file, encoding='utf-8')
        next(reader, None)  # skip the header
        test_data = map(lambda (orig, corr): orig, reader)

    with open("fix.submission.csv", "w") as submission_file:
        writer = csv.writer(submission_file, encoding='utf-8')
        writer.writerow(["Id", "Expected"])
        test_size = len(test_data)
        for (i, word) in enumerate(test_data):
            if not i % 10000:
                submission_file.flush()
                print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Processing input data:", i, "/", test_size
            fix = model.fix(word)
            writer.writerow([word, fix])
        submission_file.flush()
    submission_file.close()


if __name__ == '__main__':
    main()
