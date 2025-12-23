from typing import Callable
import networkx as nx

def rhymes_scheme(text, estimate_func: Callable):
    letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

    scores = estimate_func(text)
    rhyme_list = {
        (matche["rhyme_target"], matche["rhyme_source"] if "rhyme_source" in matche else matche["rhyme_target"])
        for score in scores
        for word in score
        for matche in word["matches"]
        if matche["score"] != 0
    }

    G = nx.Graph()
    G.add_edges_from(rhyme_list)
    cliques = nx.find_cliques(G)

    rhymes_symbol =  [
        {
            "rhyme":group,
            "letter": letters[i],
        }
        for i, group in enumerate(cliques)
    ]

    scheme = [
        [
            temp
            if (temp := ''.join(
                {
                    rhyme["letter"]
                    for match in word["matches"]
                    for rhyme in rhymes_symbol
                    if (
                        match["rhyme_target"] in rhyme["rhyme"]
                        and ("rhyme_source" not in match or match["rhyme_source"] in rhyme["rhyme"])
                    )
                }
            ))
            else '-'
            for word in line
        ]
        for line in scores
    ]
    printable_rhymes_symbol = {cell["letter"]:cell["rhyme"] for cell in rhymes_symbol}
    scheme_string = f'\n'.join(['{0:3}.\t'.format(num+1) +'|'.join(line) for num,line in enumerate(scheme)])

    return {
        "printable_rhymes_symbol":printable_rhymes_symbol,
        "scheme": scheme,
        "scheme_string":scheme_string,
    }