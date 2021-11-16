"""
Part 3: Here you should improve viterbi to use better laplace smoothing for unseen words
This should do better than baseline and your first implementation of viterbi, especially on unseen words
"""
from math import log
from collections import Counter
from collections import defaultdict


def count_occurrence(train):
    tags = Counter()
    tag_pairs = defaultdict(dict)
    tag_word_pairs = defaultdict(dict)
    hapax = Counter()
    for sentence in train:
        for word, tag in sentence:
            tags[tag] += 1
            if tag not in tag_word_pairs[word] or word not in tag_word_pairs:
                tag_word_pairs[word][tag] = 1
            else:
                tag_word_pairs[word][tag] += 1
            hapax[word] += 1
        for index in range(len(sentence) - 1):
            if sentence[index][1] not in tag_pairs[sentence[index + 1][1]] or sentence[index + 1][1] not in tag_pairs:
                tag_pairs[sentence[index + 1][1]][sentence[index][1]] = 1
            else:
                tag_pairs[sentence[index + 1][1]][sentence[index][1]] += 1
    return tags, tag_pairs, tag_word_pairs, hapax


def hapax_distribution(hapax, tag_word_pairs, tags):
    hapax_tags = Counter()
    index = 0
    for hapax_word in hapax:
        if hapax[hapax_word] == 1:
            index += 1
            (key, value), = tag_word_pairs[hapax_word].items()
            hapax_tags[key] += 1
    return_pairs = Counter()
    for tag in tags:
        return_pairs[tag] = (hapax_tags[tag] + 1) / index
    return return_pairs


def smoothed_probability(first_pairs, second_pairs, tags, smoothed):
    return_pairs = defaultdict(dict)
    for pairs_a in first_pairs:
        for pairs_b in tags:
            temp = second_pairs.get(pairs_a, {}).get(pairs_b, 0)
            return_pairs[pairs_a][pairs_b] = log((temp + smoothed) / (tags[pairs_b] + smoothed * len(first_pairs)))
    return return_pairs


def viterbi_2(train, test):
    """
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    """
    tags, tag_pairs, tag_word_pairs, hapax = count_occurrence(train)
    smoothed = 1e-5
    hapax_probability = hapax_distribution(hapax, tag_word_pairs, tags)
    transition = smoothed_probability(tags, tag_pairs, tags, smoothed)
    emission = smoothed_probability(tag_word_pairs, tag_word_pairs, tags, smoothed)
    result = []
    for sentence in test:
        trellis, trellis_path = defaultdict(dict), defaultdict(dict)
        for tag in tags:
            trellis[0][tag] = float("-inf")
            trellis_path[0][tag] = None
            if tag == "START":
                trellis[0][tag] = 1
        for index in range(1, len(sentence)):
            for tag in tags:
                max_path, max_path_previous = None, None
                for previous_tag in tags:
                    temp = log((smoothed * hapax_probability[tag]) /
                               (tags[tag] + smoothed * hapax_probability[tag] * (len(tag_word_pairs) + 1)))
                    if sentence[index] in emission and tag in emission[sentence[index]]:
                        temp = emission[sentence[index]][tag]
                    current = temp + transition[tag][previous_tag] + trellis[index - 1][previous_tag]
                    if max_path is None or current > max_path:
                        max_path = current
                        max_path_previous = previous_tag
                trellis[index][tag] = max_path
                trellis_path[index][tag] = max_path_previous
        temp = []
        index = len(sentence) - 1
        max_value = float("-inf")
        max_key = None
        for key in trellis[index].keys():
            if max_value < trellis[index][key]:
                max_value = trellis[index][key]
                max_key = key
        while max_key:
            temp.append((sentence[index], max_key))
            max_key = trellis_path[index][max_key]
            index -= 1
        result.append(temp[::-1])
    return result
