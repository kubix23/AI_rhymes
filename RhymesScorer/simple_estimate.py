from itertools import groupby, permutations, accumulate, batched
import pyphen

dic = pyphen.Pyphen(lang='pl_PL')
def simple_estimate(text: str):
    def rhyme_find(word_source, word_target):
        rhyme = ''
        rhyme_word_target_start = len(word_target["syllables"])
        rhyme_word_source_start = len(word_source["syllables"])
        for a, b in zip(word_source["syllables"][::-1], word_target["syllables"][::-1]):
            if a == b:
                rhyme = a + rhyme
                rhyme_word_target_start -= 1
                rhyme_word_source_start -= 1
            else:
                break
        return \
            {
                "rhyme": rhyme,
                "rhyme_word_source_start": word_source["syllables_num_start"] + rhyme_word_source_start,
                "rhyme_word_target_start": word_target["syllables_num_start"] + rhyme_word_target_start
            }

    def simple_score(rhyme: str,word_source, word_target):
        if word_source["word"] != word_target["word"]:
            score = len(rhyme)
        elif len(word_source["word"]) > 1 or word_source["word_num"] == 0:
            score = 1
        else:
            score = 0
        return score

    def make_simple_score(words):
        return [
            {
                "word": group[0][0]["word"],
                "matches": [
                    {
                        "word": word_target["word"],
                        "rhyme_target": rhyme["rhyme"],
                        "rhyme_len": len(rhyme["rhyme"]),
                        "score": simple_score(rhyme["rhyme"], word_source, word_target),
                    }
                    for word_source, word_target in group
                    if (rhyme := rhyme_find(word_source, word_target))["rhyme"]
                       and (
                               (
                                       word_source["line_num"] - 4 <= word_target["line_num"] <= word_source["line_num"] + 4
                                       and
                                       (
                                               rhyme["rhyme_word_source_start"] - 1 <= rhyme[
                                           "rhyme_word_target_start"] <= rhyme["rhyme_word_source_start"] + 1
                                               or
                                               (
                                                       word_source["word_num"] == word_source["word_in_line"]
                                                       and
                                                       word_target["word_num"] == word_target["word_in_line"]
                                               )
                                       )
                               )
                               or
                               (word_source["line_num"] == word_target["line_num"])
                       )
                ]
            }
            for group in  batched(permutations(words, 2), len(words)-1)
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
    words_score = make_simple_score(sum(words,[]))
    return [words_score[words_line_num[i]:words_line_num[i+1]] for i in range(len(words))]