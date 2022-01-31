import pickle as pkl
import numpy as np
from tqdm import tqdm
import os

DICTIONARY = np.load('bitword_dictionary.npy')
CANDIDATES = np.load('bitword_candidates.npy')

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


class State:
    def __init__(self, word):
        self.word = word
        self.candidates = CANDIDATES.copy()
        self.hints = []

    def score(self):
        return len(self.candidates)

    def guess(self, guess):
        hints = generate_hints(self.word, guess)

        green = [(i,gc) for i,(h,gc) in enumerate(zip(hints,guess)) if h==1]
        yellow = [(i,gc) for i,(h,gc) in enumerate(zip(hints,guess)) if h==0]
        grey = [(i,gc) for i,(h,gc) in enumerate(zip(hints,guess)) if h==-1]

        exact_counts = set()
        exact_chars = set()
        grey_to_yellow = set()
        for i,gc in grey:
            if gc in self.word:
                exact_counts.add((gc,(self.word==gc).sum()))
                exact_chars.add(gc)
                grey_to_yellow.add((i,gc))
        min_counts = set()
        for i,gc in green+yellow:
            if gc not in exact_chars:
                min_counts.add((gc,(guess==gc).sum()))
        for s in grey_to_yellow:
            grey.remove(s)
            yellow.append(s)

        keep = np.arange(len(self.candidates),dtype=int)
        for j,c in green:
            keep = keep[self.candidates[keep,j]==c]
        for j,c in yellow:
            keep = keep[self.candidates[keep,j]!=c]
        for c,n in exact_counts:
            keep = keep[(self.candidates[keep]==c).sum(axis=1)==n]
        for c,n in min_counts:
            keep = keep[(self.candidates[keep]==c).sum(axis=1)>=n]
        for j,c in grey:
            keep = keep[np.all(self.candidates[keep]!=c,axis=1)]

        return keep

def main():
    if os.path.exists('first_move.pkl'):
        with open('first_move.pkl', 'rb') as f:
            guess_stats = pkl.load(f)['guess_stats']
    else:
        guess_stats = {decode(w): {'total': 0, 'min': float('inf'), 'max': -float('inf'), 'num': 0} for w in DICTIONARY}
        word_stats = dict()
        for i, word in tqdm(enumerate(CANDIDATES), total=len(CANDIDATES)):
            s = State(word)
            tot = 0
            mn = float('inf')
            mx = -float('inf')
            for guess in DICTIONARY:
                remaining = s.guess(guess).size
                tot += remaining
                mn = min(mn, remaining)
                mx = max(mx, remaining)

                dw = decode(guess)
                guess_stats[dw]['total'] += remaining
                guess_stats[dw]['min'] = min(guess_stats[dw]['min'], remaining)
                guess_stats[dw]['max'] = max(guess_stats[dw]['max'], remaining)
                guess_stats[dw]['num'] = i + 1

            word_stats[decode(word)] = {'total': tot, 'min': mn, 'max': mx}
        print('saving results to first_move.pkl')
        with open('first_move.pkl', 'wb') as f:
            pkl.dump({'word_stats': word_stats, 'guess_stats': guess_stats}, f)

    # best word is one that gives lowest number of candidates, tie broken by lowest average number of candidates
    best_starting = min(guess_stats, key=lambda k: (guess_stats[k]['max'], guess_stats[k]['total']))
    print('Best starting word is:', best_starting)  # Best starting word is: raise


if __name__ == "__main__":
    main()
