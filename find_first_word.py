import pickle as pkl
import numpy as np
from tqdm import tqdm
import os
from collections import Counter

DICTIONARY = np.load('bitword_dictionary.npy')
CANDIDATES = np.load('bitword_dictionary.npy')


def encode(word):
    return np.array([1 << (ord(v)-ord('a')) for v in word], dtype=np.uint32)


def decode(bitword):
    return ''.join(chr(ord('a') + int(n).bit_length() - 1) for n in bitword)


def generate_hints(word, guess):
    hints = np.array([2 * int(wc == gc) - 1 for wc, gc in zip(word, guess)], dtype=np.int32)
    idx = [i for i, (wc, gc) in enumerate(zip(word, guess)) if wc != gc]
    rem = [word[i] for i in idx]
    for i in idx:
        gc = guess[i]
        if gc in rem:
            rem.remove(gc)
            hints[i] = 0
    return hints


def generate_all_hints(guess, candidates):
    green = (guess[None, :] == candidates).astype(int)
    not_grey = np.zeros(green.shape, dtype=int)

    num_rem = dict()
    for c,n in Counter(guess).items():
        if n > 1:
            matches = (CANDIDATES == c)
            num_correct = np.logical_and(matches, green).sum(axis=1)
            num_in = matches.sum(axis=1)
            num_rem[c] = num_in - num_correct

    for i,c in enumerate(guess):
        if c in num_rem:
            not_grey[:, i] = np.logical_or(num_rem[c] > 0, green[:,i])
            num_rem[c] -= (1 - green[:, i])
        else:
            not_grey[:, i] = np.any(CANDIDATES == c, axis=1)

    hints = not_grey + green - 1
    return hints


def count_remaining_candidates(guess, hints):
    green = [(i, gc) for i, (h, gc) in enumerate(zip(hints, guess)) if h == 1]
    yellow = [(i, gc) for i, (h, gc) in enumerate(zip(hints, guess)) if h == 0]
    grey = [(i, gc) for i, (h, gc) in enumerate(zip(hints, guess)) if h == -1]

    exact_counts = set()
    exact_chars = set()
    grey_to_yellow = set()
    for i, gc in grey:
        count = sum(c==gc for _,c in yellow+green)
        if count > 0:
            exact_counts.add((gc, count))
            exact_chars.add(gc)
            grey_to_yellow.add((i, gc))
    min_counts = set()
    for i, gc in green + yellow:
        if gc not in exact_chars:
            min_counts.add((gc, (guess == gc).sum()))
    for s in grey_to_yellow:
        grey.remove(s)
        yellow.append(s)

    match = np.arange(len(CANDIDATES), dtype=int)
    for j, c in green:
        match = match[CANDIDATES[match, j] == c]
    for j, c in yellow:
        match = match[CANDIDATES[match, j] != c]
    for c, n in exact_counts:
        match = match[(CANDIDATES[match] == c).sum(axis=1) == n]
    for c, n in min_counts:
        match = match[(CANDIDATES[match] == c).sum(axis=1) >= n]
    for j, c in grey:
        match = match[np.all(CANDIDATES[match] != c, axis=1)]

    return match.size


def main():
    if os.path.exists('first_move_12k.pkl'):
        with open('first_move_12k.pkl', 'rb') as f:
            guess_stats = pkl.load(f)
    else:
        from multiprocessing import Pool
        guess_stats = dict()
        for guess in tqdm(DICTIONARY):
            stats = {'total': 0, 'min': float('inf'), 'max': -float('inf'), 'num': 0}
            for hints, num_repeats in zip(*np.unique(generate_all_hints(guess, CANDIDATES), return_counts=True, axis=0)):
                num_remaining = count_remaining_candidates(guess, hints)
                stats['min'] = min(stats['min'], num_remaining)
                stats['max'] = max(stats['max'], num_remaining)
                stats['total'] += num_remaining * num_repeats
                stats['num'] += num_repeats
            guess_stats[decode(guess)] = stats
        print('saving results to first_move_12k.pkl')
        with open('first_move_12k.pkl', 'wb') as f:
            pkl.dump(guess_stats, f)

    # best word is one that gives lowest number of candidates, tie broken by lowest average number of candidates
    best_starting = min(guess_stats, key=lambda k: (guess_stats[k]['max'], guess_stats[k]['total']))
    print('Best starting word is:', best_starting)  # Best starting word is: raise


if __name__ == "__main__":
    main()
