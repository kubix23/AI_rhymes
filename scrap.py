"""
Download and save Polish poetry from Wolne Lektury.

This script fetches a list of poetry books from the Wolne Lektury API,
downloads the text of each poem, cleans it by removing headers and
punctuation, and saves each poem as a .txt file. Finally, all poems are
compressed into a single ZIP archive.

Modules required:
- os
- re
- string
- requests
- zipfile
- tqdm
"""

import os
import re

import requests
import zipfile
from tqdm import tqdm

API = "https://wolnelektury.pl/api/books"
OUT = "polish_poetry"
ZIP = "polish_poetry.zip"
number = 4000


os.makedirs(OUT, exist_ok=True)

# 1) pobierz listę wierszy
resp = requests.get(API)
resp.raise_for_status()
data = [i for i in resp.json() if i.get("kind") == "Liryka"]

print(f"Znaleziono: {len(data)} wierszy")

# 2) pobieranie każdego .txt
poems = []

for item in tqdm(data[:number]):
    title = item.get("title", "unknown").replace("/", "_")
    authors = item.get("authors", [])
    author = author[0]["name"] if authors else item.get("author")
    txt_url = requests.get(item.get("href")).json().get("txt")

    if not txt_url:
        continue

    try:
        txt = requests.get(txt_url).text
        filename = f"{author} - {title}.txt"
        path = os.path.join(OUT, filename)

        txt2 = txt.split("-----")[0].split("\r\n\r\n\r\n\r\n",1)[1]
        txt3 = '\n'.join([re.sub(r'[^\w\s]', '', i.lower().strip())  for i in txt2.splitlines()])

        with open(path, "w", encoding="utf-8") as f:
            f.write(txt3)

        poems.append(path)

    except Exception as e:
        print("ERR:", e)

# 3) ZIP
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for p in poems:
        z.write(p, arcname=os.path.basename(p))

print("Gotowe →", ZIP)
