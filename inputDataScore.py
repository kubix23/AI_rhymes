"""
Compute and save scores for Polish poems in parallel.

This script reads poem files from a specified directory, calculates a
score for each poem using the `score` function, and writes the results
to an output file. Processing is done in parallel using
`ProcessPoolExecutor`.

Modules required:
- os
- concurrent.futures
- tqdm
- score (custom scoring module)
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from RhymesScorer.score import score

POETRY_DIR = "E:/Projects/IUI/Polish_poetry/"
OUT_FILE = "E:/Projects/IUI/scoresPoetry"

def myscore(x):
    return sum(sum(score(x, type="advanced"),[]))
num = 0

def process_file(file):
    path = os.path.join(POETRY_DIR, file)
    table = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                table.append(line.strip())
            if len(table) == 40:
                break

    if len(table) > 5:
        return file, myscore(' \n'.join(table))
    return None


if __name__ == "__main__":
    files = os.listdir(POETRY_DIR)

    with open(OUT_FILE, "w", encoding="utf-8") as f2, \
            ProcessPoolExecutor() as executor:

        futures = [executor.submit(process_file, f) for f in files]
        num = 0
        for future in tqdm(as_completed(futures), total=len(futures)):
            num += 1
            result = future.result()
            if result:
                file, score = result
                f2.write(f"{file}|{score}\n")
            if num%100:
                f2.flush()
