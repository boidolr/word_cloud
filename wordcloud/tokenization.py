from __future__ import division
from itertools import tee
from operator import itemgetter
from collections import defaultdict, Counter
from math import log


def l(k, n, x):  # noqa: E741, E743
    return log(max(x, 1e-10)) * k + log(max(1 - x, 1e-10)) * (n - k)


def score(count_bigram, count1, count2, n_words):
    """Collocation score"""
    if n_words <= count1 or n_words <= count2:
        return 0
    N = n_words
    c12 = count_bigram
    c1 = count1
    c2 = count2
    p = c2 / N
    p1 = c12 / c1
    p2 = (c2 - c12) / (N - c1)
    score = (l(c12, c1, p) + l(c2 - c12, N - c1, p)
             - l(c12, c1, p1) - l(c2 - c12, N - c1, p2))
    return -2 * score


def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def unigrams_and_bigrams(words, stopwords, normalize_plurals=True, collocation_threshold=30):
    stopwords_lower = set(w.lower() for w in stopwords)
    words_lower = [w.lower() for w in words]
    
    unigrams = []
    bigrams = []
    for i, word in enumerate(words):
        if words_lower[i] not in stopwords_lower:
            unigrams.append(word)
            if i > 0 and words_lower[i-1] not in stopwords_lower:
                bigrams.append((words[i-1], word))
    
    n_words = len(unigrams)
    counts_unigrams, standard_form = process_tokens(
        unigrams, normalize_plurals=normalize_plurals)
    counts_bigrams, standard_form_bigrams = process_tokens(
        [" ".join(bigram) for bigram in bigrams],
        normalize_plurals=normalize_plurals)
    orig_counts = counts_unigrams.copy()

    for bigram_string, count in counts_bigrams.items():
        bigram = tuple(bigram_string.split(" "))
        word1 = standard_form[bigram[0].lower()]
        word2 = standard_form[bigram[1].lower()]

        collocation_score = score(count, orig_counts[word1], orig_counts[word2], n_words)
        if collocation_score > collocation_threshold:
            counts_unigrams[word1] -= counts_bigrams[bigram_string]
            counts_unigrams[word2] -= counts_bigrams[bigram_string]
            counts_unigrams[bigram_string] = counts_bigrams[bigram_string]
    for word, count in list(counts_unigrams.items()):
        if count <= 0:
            del counts_unigrams[word]
    return counts_unigrams


def process_tokens(words, normalize_plurals=True):
    """Normalize cases and remove plurals.

    Each word is represented by the most common case.
    If a word appears with an "s" on the end and without an "s" on the end,
    the version with "s" is assumed to be a plural and merged with the
    version without "s" (except if the word ends with "ss").

    Parameters
    ----------
    words : iterable of strings
        Words to count.

    normalize_plurals : bool, default=True
        Whether to try and detect plurals and remove trailing "s".

    Returns
    -------
    counts : dict from string to int
        Counts for each unique word, with cases represented by the most common
        case, and plurals removed.

    standard_forms : dict from string to string
        For each lower-case word the standard capitalization.
    """
    case_counts = Counter()
    lower_to_cases = defaultdict(Counter)
    
    for word in words:
        word_lower = word.lower()
        case_counts[word] += 1
        lower_to_cases[word_lower][word] += 1
    
    if normalize_plurals:
        merged_plurals = {}
        keys_to_process = list(lower_to_cases.keys())
        for key in keys_to_process:
            if key.endswith('s') and not key.endswith("ss"):
                key_singular = key[:-1]
                if key_singular in lower_to_cases:
                    for word, count in lower_to_cases[key].items():
                        singular = word[:-1]
                        lower_to_cases[key_singular][singular] += count
                    merged_plurals[key] = key_singular
                    del lower_to_cases[key]
    
    fused_cases = {}
    standard_cases = {}
    for word_lower, case_dict in lower_to_cases.items():
        first = case_dict.most_common(1)[0][0]
        fused_cases[first] = sum(case_dict.values())
        standard_cases[word_lower] = first
    
    if normalize_plurals:
        for plural, singular in merged_plurals.items():
            standard_cases[plural] = standard_cases[singular.lower()]
    
    return fused_cases, standard_cases
