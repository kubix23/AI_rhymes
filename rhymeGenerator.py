"""
Generate and save Polish rhymes.

This script reads a pronunciation dictionary from a CSV file, groups words
by their rhyme endings, randomly selects a fixed number of words from each
group, and saves the resulting rhymes to a CSV file.

Modules required:
- random
- pandas
- pyphen
- epitran
- tqdm
"""

import pandas as pd
import pyphen
import epitran
from tqdm import tqdm
dic = pyphen.Pyphen(lang='pl_PL')
epi = epitran.Epitran('pol-Latn')

if __name__ == '__main__':
    N = 6
    pronDict = pd.read_csv("E:\Projects\IUI\dictionary.csv", header=None, encoding="utf-8")
    source = pronDict.get(0).str.strip()
    target = pronDict.get(1).str.strip()
    pronDict = pd.DataFrame({"source": source, "target": target})

    rhymes = {
        rhyme[0]:rhyme[1].sample(N, )["source"].to_list()
        for rhyme in tqdm(pronDict.groupby("target"), desc="rhymes") if len(rhyme[1]) > N
    }
    pd.DataFrame.from_dict(rhymes).T.to_csv("E:/Projects/IUI/rhymes.csv", sep='|', encoding='utf-8', header=False, index=False)
