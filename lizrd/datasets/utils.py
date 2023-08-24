import random


def get_random_chunk(ids, length, rng=None):
    if rng == None:
        rng = random
    beginning = rng.randint(0, len(ids) - length)
    return ids[beginning : beginning + length]
