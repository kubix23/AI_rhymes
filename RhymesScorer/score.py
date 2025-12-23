def score(text, type = '', threshold = 0):
    """
    Compute rhyme scores for a given Polish text.

    This function calculates scores for rhymes in the input text using
    either a simple or advanced estimation method.

    Args:
        text (str): Input text (poem or lines) to score.
        type (str, optional): Type of scoring method.
            - 'advanced': uses `advanced_estimate` for more detailed scoring.
            - any other value (default): uses `simple_estimate`.
        threshold (float, optional): Threshold parameter for `advanced_estimate`.
            Default is 0. Only used if type='advanced'. It means the phonetic
            distance of a word that we consider a rhyme.

    Returns:
        list[list[float]]: A nested list of scores. Outer list corresponds
        to lines in the text, inner lists correspond to words in each line,
        with each element being the sum of scores of matched rhymes.
    """
    match type.lower():
        case 'advanced':
            from RhymesScorer.advanced_estimate import advanced_estimate
            return [
                [
                    sum([i["score"] for i in word["matches"]])
                    for word in line
                ]
                for line in advanced_estimate(text, threshold)
            ]
        case _:
            from RhymesScorer.simple_estimate import simple_estimate
            return [
                [
                    sum([i["score"] for i in word["matches"]])
                    for word in line
                ]
                for line in simple_estimate(text)
            ]