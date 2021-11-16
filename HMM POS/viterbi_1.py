"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""
from math import log
from collections import Counter
from collections import defaultdict


def count_occurrence(train):
    tags = Counter()
    tag_pairs = defaultdict(dict)
    tag_word_pairs = defaultdict(dict)
    for sentence in train:
        for word, tag in sentence:
            tags[tag] += 1
            if tag not in tag_word_pairs[word] or word not in tag_word_pairs:
                tag_word_pairs[word][tag] = 1
            else:
                tag_word_pairs[word][tag] += 1
        for index in range(len(sentence) - 1):
            if sentence[index][1] not in tag_pairs[sentence[index + 1][1]] or sentence[index + 1][1] not in tag_pairs:
                tag_pairs[sentence[index + 1][1]][sentence[index][1]] = 1
            else:
                tag_pairs[sentence[index + 1][1]][sentence[index][1]] += 1
    return tags, tag_pairs, tag_word_pairs


def smoothed_probability(first_pairs, second_pairs, tags, smoothed):
    return_pairs = defaultdict(dict)
    for pairs_a in first_pairs:
        for pairs_b in tags:
            temp = second_pairs.get(pairs_a, {}).get(pairs_b, 0)
            return_pairs[pairs_a][pairs_b] = log((temp + smoothed) / (tags[pairs_b] + smoothed * len(first_pairs)))
    return return_pairs


def viterbi_1(train, test):
    """
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    """
    tags, tag_pairs, tag_word_pairs = count_occurrence(train)
    smoothed = 0.001
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
                    temp = log(smoothed / (tags[tag] + smoothed * len(tag_word_pairs)))
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
