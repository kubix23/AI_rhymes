# AI rhymes

---

## Polska wersja

### Wstęp

Celem projektu jest stworzenie Ai które, na podstawie podanego
podanego słowa będzie w stanie dobrać do niego rym.

### Ocena rymu

Do oceny rymu wykorzystywana jest funkcja `score`. Ma ona dwie
wersje. Obie zwracają tablicę tablic, gdzie każda z liczb odpowiada
słowu w danym wierszu. Wielkość tej wartości zależy od wybranej wersji.
Dla wersji podstawowej jest to suma wszystkich pasujących slab.
Dla wersji zaawansowanej (po podanie `type='advance'`) 
zwracana jest wartość z przedziału od 0-10 w zależności od odległości 
fonetycznej slab (pod warunkiem, że odległość jest mniejsza niż `threshold`).
Dalsze informacje i szczegóły funkcji oceny znajdują się pod linkiem 
https://github.com/kubix23/Rhyme_detector.

### Testowany model

Pierwszy wybrany model bazował na angielskim modelu BERT, ale
przerobiony na język polski. Model bazuje na podaniu na wejście
całego ztokenizowanego tekstu z umieszczoną specjalną flagą w miejscu
poszukiwanego słowo. Następnie zwraca cały tekst wypełniony
specjalnymi flagami poza miejscami gdzie wstawiono flagi. 
Najpierw uczony był w nadzorowany sposób zastępując jedno słowo
z danych wejściowych maską. Następnie na takiej samej zasadzie
uczony był ten model przez uczenie nadzorowane. Nagradzany był
za wyższą wartość oceny rymowania się tekstu.

### Screeny z uczenia modelu
1. Przetrenowany na wierszach, aby podawał bardziej słownictwo wierszów.
![](Docs\Images\Zrzut ekranu 2025-12-15 200947.png)
2. Wprowadzono  batch baseline, dodanie entropii, Normalizuj przez liczbę masków,
zmniejszenie learning rate, zmniejszenie danych wejściowych.
![](Docs\Images\Zrzut ekranu 2025-12-15 224142.png)
3. Próba na samych rymach, po przetrenowaniu nadzorowanym.
![](Docs\Images\Zrzut ekranu 2025-12-16 140209.png)

### Zadania

- [x] **Zebranie danych treningowych** - zbiór wierszy polskich, oraz zbiór 6 słów
w każdej linijce rymujących się ze sobą
- [x] **Wybranie modelu** - model bazujący na BERT MLM
- [x] **Przetestowanie modelu** - model przy wielu różnych kombinacjach nie 
może osiągnąć oryginalnej oceny bez zamaskowanego słowa.
- [ ] **Wybranie nowego modelu**
- [ ] **Przetestowanie modelu**
- [ ] **Zmiana reprezentacji danych wejściowych**
- [ ] **Poszukanie jeszcze innych alternatyw**

### Możliwe rozwiązania

1. **Nowy model dla LLM** \
    Chodzi o spróbowanie wytrenować modele typu GPT2 czy 
    inne LLM najpierw w sposób nadzorowany, aby nauczyć go mniej więcej szukać słów
    potem przez wzmocnienie wytrenować w nim zdolność wykrywania rymów.
2. **Zmiana reprezentacji danych** \
    Nauczyciel wspomniał o zmianie reprezentacji danych jako zbiór przedstawiający 
    pozycję litery i numer jej przypisany, zmniejszyć potem wymiar tych danych
    jakąś warstwą i wtedy spróbować szkolić sieć. Może tu pojawić
    się problem z tym, że dane wyjściowe muszą być potem czytelne/string. 
    Ponieważ w takim formacie oczekiwane jest wejście funkcji oceny.
3. **Analiza innych rozwiązań**

---

## English version

### Introduction

The aim of the project is to create an AI that, based on a given word, 
can generate a suitable rhyme for it.

### Rhyme Evaluation

The `score` function is used to evaluate rhymes. It has two versions. 
Both return an array of arrays, where each number corresponds to a word 
in a given line. The magnitude of this value depends on the chosen version.  
For the basic version, it is the sum of all matching syllables.  
For the advanced version (by passing `type='advance'`), the returned value 
ranges from 0–10 depending on the phonetic distance of syllables 
(provided the distance is less than `threshold`).  
Further details about the evaluation function can be found at  
https://github.com/kubix23/Rhyme_detector.

### Tested Model

The first chosen model was based on the English BERT model but adapted for Polish.  
The model takes the entire tokenized text as input, with a special flag inserted 
at the position of the target word. It then returns the full text filled with 
special flags except for the positions where the flags were inserted.  
Initially, it was trained in a supervised manner by replacing one word in 
the input data with a mask. Then, using a similar approach, the model was 
trained with reinforcement learning, being rewarded for higher rhyme evaluation scores.

### Model Training Screenshots
1. Trained on poems to provide more vocabulary from the lines.  
![](Docs\Images\Zrzut ekranu 2025-12-15 200947.png)

2. Introduced batch baseline, added entropy, normalized by the number of masks,  
reduced learning rate, and reduced input data.  
![](Docs\Images\Zrzut ekranu 2025-12-15 224142.png)

3. Attempt on rhymes only, after supervised pretraining.  
![](Docs\Images\Zrzut ekranu 2025-12-16 140209.png)

### Tasks

- [x] **Collect training data** – a corpus of Polish lines, and a set of 6 words 
per line that rhyme with each other
- [x] **Select a model** – a model based on BERT MLM
- [x] **Test the model** – the model should not reach the original evaluation 
score with many different combinations if the target word is masked
- [ ] **Select a new model**
- [ ] **Test the new model**
- [ ] **Change the input data representation**
- [ ] **Explore other alternatives**

### Possible Solutions

1. **New model for LLM**  
   Try training GPT-2 or other LLM models first in a supervised way to roughly 
   teach them to search for words, then use reinforcement learning to train 
   their ability to detect rhymes.

2. **Change the input data representation**  
   The instructor suggested representing data as a set showing the position 
   of each letter and its assigned number, then reducing the dimensionality 
   with some layer, and then training the network. A potential issue is that 
   the output data must still be readable/string, as the evaluation function 
   expects input in this format.

3. **Analyze other solutions**

---

## Links

- ...
- ... 
