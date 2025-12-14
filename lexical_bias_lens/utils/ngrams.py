from typing import List, Any


def get_ngrams(tokens: List[Any], min_n: int = 1, max_n: int = 1) -> List[tuple]:
    """
    This function generates n-grams from a list of tokens.
    Returns a list of n-grams as tuples.
    """
    ngrams = []
    for n in range(min_n, max_n + 1):
        # Ensure n is not greater than the number of tokens
        if n > len(tokens):
            break
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            ngrams.append(ngram)
    return ngrams