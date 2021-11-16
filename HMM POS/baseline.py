"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""


def baseline(train, test):
    """
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    """
    train_table = {}
    tags_table = {}
    for sentence in train:
        for word, tag in sentence:
            if not train_table.get(word):
                train_table[word] = {tag: 1}
            elif train_table[word].get(tag):
                train_table[word][tag] += 1
            else:
                train_table[word][tag] = 1
            if tags_table.get(tag):
                tags_table[tag] += 1
            else:
                tags_table[tag] = 1
    ret = []
    for sentence in test:
        temp_sentence = []
        for word in sentence:
            if train_table.get(word):
                tag = max(train_table[word], key=train_table[word].get)
            else:
                tag = max(tags_table, key=tags_table.get)
            temp_sentence.append((word, tag))
        ret.append(temp_sentence)
    return ret
