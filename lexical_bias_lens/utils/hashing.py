import hashlib


def hash_sample(sample: list) -> str:
    """
    Generates a unique hash for a given sample (list of tokens).
    """
    sample_str = ' '.join(map(str, sample))
    return hashlib.sha256(sample_str.encode('utf-8')).hexdigest()