import sys
from itertools import groupby, permutations, accumulate, batched

import pyphen
import epitran
import panphon.distance
epi = epitran.Epitran('pol-Latn')
dst = panphon.distance.Distance()
dic = pyphen.Pyphen(lang='pl_PL')

def advanced_estimate(text: str, threshold = 0):
    def pronunciation_distance(war1, war2):
        fon1, fon2 = epi.transliterate(war1), epi.transliterate(war2)
        return dst.weighted_feature_edit_distance(fon1, fon2)

    def rhyme_find(word_source, word_target):
        rhyme_source = ''
        rhyme_target = ''
        rhyme_word_target_start = len(word_target["syllables"])
        rhyme_word_source_start = len(word_source["syllables"])
        counter = 0
        for a, b in zip(word_source["syllables"][::-1], word_target["syllables"][::-1]):
            if pronunciation_distance(ra := a + rhyme_source,rb := b + rhyme_target) <= threshold:
                rhyme_source = ra
                rhyme_target = rb
                counter += 1
            else:
                break

        return \
            {
                "rhyme_source": rhyme_source,
                "rhyme_target": rhyme_target,
                "rhyme_word_source_start": word_source["syllables_num_start"] + rhyme_word_source_start - counter,
                "rhyme_word_target_start": word_target["syllables_num_start"] + rhyme_word_target_start - counter
            }

    def advanced_score(rhyme, word_source, word_target):
        mul = pronunciation_distance(rhyme["rhyme_source"], rhyme["rhyme_target"])/(threshold + sys.float_info.epsilon)
        if word_source["word"] != word_target["word"]:
            score = 10 - 10*mul
        elif (len(word_source["word"]) > 1
              or (word_source["word_num"] == 0 and word_target["word_num"] == 0)):
            score = 1
        else:
            score = 0
        return score

    def make_advanced_score(words):
        return [
            {
                "word": group[0][0]["word"],
                "matches": [
                    {
                        "word": word_target["word"],
                        "rhyme_source": rhyme["rhyme_source"],
                        "rhyme_target": rhyme["rhyme_target"],
                        "distance_word": pronunciation_distance(word_source["word"], word_target["word"]),
                        "distance_rhyme": pronunciation_distance(rhyme["rhyme_source"], rhyme["rhyme_target"]),
                        "score": advanced_score(rhyme, word_source, word_target),
                    }
                    for word_source, word_target in group
                    if (
                           word_source["line_num"] == word_target["line_num"]
                           and
                           (rhyme := rhyme_find(word_source, word_target))["rhyme_target"]
                       )
                       or
                       (
                           word_source["line_num"] - 4 <= word_target["line_num"] <= word_source["line_num"] + 4
                           and
                           (rhyme := rhyme_find(word_source, word_target))["rhyme_target"]
                           and
                           (
                               rhyme["rhyme_word_source_start"] - 1
                               <= rhyme["rhyme_word_target_start"] <=
                               rhyme["rhyme_word_source_start"] + 1
                               or
                               (
                                   word_source["word_num"] == word_source["word_in_line"]
                                   and
                                   word_target["word_num"] == word_target["word_in_line"]
                               )
                           )
                       )
                ]
            }
            for group in batched(permutations(words, 2), len(words)-1)
        ]

    words = [
        [
            {
                "word": word,
                "line_num":line_num,
                "word_in_line":len(line.split())-1,
                "word_num":word_num,
                "syllables": dic.inserted(word).split('-'),
                "syllables_num_start":sum([len(dic.inserted(i).split('-')) for i in line.split()[:word_num]]),
            }
            for word_num,word in enumerate(line.split())
        ]
        for line_num,line in enumerate(text.splitlines())
        if line
    ]
    words_line_len = [len(i) for i in words]
    words_line_num = sum([[0],list(accumulate(words_line_len))],[])
    words_score = make_advanced_score(sum(words,[]))
    return [words_score[words_line_num[i]:words_line_num[i+1]] for i in range(len(words))]