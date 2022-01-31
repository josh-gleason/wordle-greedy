import numpy as np
from tqdm import tqdm
import pickle as pkl
import sys


def _generate_hints(word, guess):
    hints = np.array([2 * int(wc == gc) - 1 for wc, gc in zip(word, guess)], dtype=np.int32)
    idx = [i for i, (wc, gc) in enumerate(zip(word, guess)) if wc != gc]
    rem = [word[i] for i in idx]
    for i in idx:
        gc = guess[i]
        if gc in rem:
            rem.remove(gc)
            hints[i] = 0
    return hints


def _encode(word):
    return np.array([1 << (ord(v) - ord('a')) for v in word], dtype=np.int32)


def _decode(bitword):
    return ''.join(chr(ord('a') + int(n).bit_length() - 1) for n in bitword)


def hints_to_str(hints):
    yellow = 'Y'
    green = 'G'
    grey = 'X'
    chars = [grey if h == -1 else yellow if h == 0 else green for h in hints]
    return f'[{",".join(chars)}]'


class Engine:
    def __init__(self):
        self.candidates = np.load('bitword_candidates.npy')
        self.dictionary = np.load('bitword_dictionary.npy')
        self.guesses = 0
        self.guess_list = []
        self.hint_list = []
        with open('lookup_table_raise.pkl', 'rb') as f:
            self.lookup_table = pkl.load(f)


    def apply_guess(self, guess, hints):
        keep_idx = self._get_match_indices(_encode(guess), hints)
        self.candidates = self.candidates[keep_idx]
        self.guesses += 1
        self.guess_list.append(guess)
        self.hint_list.append(tuple(hints.tolist()))

    def get_best_guess(self):
        d = self.lookup_table
        found = True
        for guess, hint in zip(self.guess_list, self.hint_list):
            if guess in d and hint in d[guess]:
                d = d[guess][hint]
            else:
                found = False
                break
        if found:
            for k in d:
                return k

        if len(self.candidates) < 3:
            return _decode(self.candidates[0])

        min_max_match = float('inf')
        min_tot_match = float('inf')
        min_max_guess = []
        for guess in tqdm(self.dictionary):
            tot_match = 0
            max_match = -float('inf')
            for word in self.candidates:
                hints = _generate_hints(word, guess)
                num_match = self._get_match_indices(guess, hints).size
                tot_match += num_match
                max_match = max(max_match, num_match)

            if max_match < min_max_match:
                min_max_guess = [(tot_match, guess)]
                min_max_match = max_match
            elif max_match == min_max_match:
                min_max_guess.append((tot_match, guess))

            if tot_match < min_tot_match:
                min_tot_match = tot_match

        guess_options = min_max_guess
        guess_options = sorted(guess_options, key=lambda t: t[0])

        for secondary, guess in guess_options:
            if guess in self.candidates:
                return _decode(guess)
        return _decode(guess_options[0][1])

    def _get_match_indices(self, guess, hints):
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

        match = np.arange(len(self.candidates), dtype=int)
        for j, c in green:
            match = match[self.candidates[match, j] == c]
        for j, c in yellow:
            match = match[self.candidates[match, j] != c]
        for c, n in exact_counts:
            match = match[(self.candidates[match] == c).sum(axis=1) == n]
        for c, n in min_counts:
            match = match[(self.candidates[match] == c).sum(axis=1) >= n]
        for j, c in grey:
            match = match[np.all(self.candidates[match] != c, axis=1)]

        return match


class Game:
    def __init__(self, word):
        self.word = word
        self.guesses = 0

    def guess(self, guess):
        self.guesses += 1
        hints = _generate_hints(self.word, guess)
        return hints


def count_guesses(word):
    game = Game(word)
    eng = Engine()

    guesses = 0
    hints = np.zeros(5, dtype=int)
    while not np.all(hints == 1):
        guess = eng.get_best_guess()
        hints = game.guess(guess)
        eng.apply_guess(guess, hints)
        guesses += 1
    return guesses


def build_lookup_table():
    word_map = {'raise': {}}
    TOTAL_HINTS = len({tuple(_generate_hints(w, 'raise').tolist()) for w in map(_decode,Engine().candidates) if w!='raise'})

    def recursive(args, d, level=0):
        temp_eng = Engine()
        for g,h in args:
            temp_eng.apply_guess(g,h)
        candidates = list(map(_decode,temp_eng.candidates))

        for guess in d:
            for word in candidates:
                if word == guess:
                    continue
                hints = _generate_hints(word, guess)
                k = tuple(hints.tolist())
                if k not in d[guess]:
                    eng = Engine()
                    for g,h in args:
                        eng.apply_guess(g,h)
                    eng.apply_guess(guess, hints)
                    if level == 0:
                        print(f'Processing hint {(len(d[guess])+1)}/{TOTAL_HINTS}', file=sys.stderr)
                    best = eng.get_best_guess()
                    d[guess][k] = {best: {}}
                    recursive(args + [(guess, hints)], d[guess][k], level+1)

    recursive([], word_map)

    with open('lookup_table_raise.pkl', 'wb') as f:
        pkl.dump(word_map, f)


def interactive_helper():
    import re
    eng = Engine()
    best = eng.get_best_guess()
    turns = 0
    while True:
        turns += 1
        print('Use the word:', best.upper())
        clues = input('Enter resulting clues ([XYG]{5}): ')
        while not re.match(r'^[XYG]{5}$', clues):
            print('Invalid clues, please enter 5 characters without spaces from X,Y,or G.')
            clues = input('Enter resulting clues ([XYG]{5}): ')
        if clues == 'GGGGG':
            break
        hint = np.array([-1 if c=='X' else 0 if c=='Y' else 1 for c in clues])
        eng.apply_guess(best, hint)
        best = eng.get_best_guess()
    print(f'Congratulations you won in {turns} turns!')


if __name__ == "__main__":
    # count = [count_guesses(word) for word in map(_decode,Engine().candidates)]
    # print('max guesses =',max(count))
    # print('avg guesses =',sum(count)/len(count))

    interactive_helper()

    # build_lookup_table()