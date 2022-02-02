import numpy as np
from tqdm import tqdm
import pickle as pkl
import sys
from collections import Counter


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


def _generate_all_hints(guess, candidates):
    green = (guess[None, :] == candidates).astype(int)
    not_grey = np.zeros(green.shape, dtype=int)

    num_rem = dict()
    for c,n in Counter(guess).items():
        if n > 1:
            matches = (candidates == c)
            num_correct = np.logical_and(matches, green).sum(axis=1)
            num_in = matches.sum(axis=1)
            num_rem[c] = num_in - num_correct

    for i,c in enumerate(guess):
        if c in num_rem:
            not_grey[:, i] = np.logical_or(num_rem[c] > 0, green[:,i])
            num_rem[c] -= (1 - green[:, i])
        else:
            not_grey[:, i] = np.any(candidates == c, axis=1)

    hints = not_grey + green - 1
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
        self.candidates = np.load('bitword_dictionary.npy')
        self.dictionary = np.load('bitword_dictionary.npy')
        self.guesses = 0
        self.guess_list = []
        self.hint_list = []
        self.lookup_table = dict()
        with open('lookup_table_par.pkl', 'rb') as f:
            self.lookup_table = pkl.load(f)

    def apply_guess(self, guess, hints):
        keep_idx = self._get_match_indices(_encode(guess), hints)
        self.candidates = self.candidates[keep_idx]
        self.guesses += 1
        self.guess_list.append(guess)
        self.hint_list.append(tuple(hints.tolist()))

    def get_best_guess(self, pbar_loc=0):
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
        for guess in tqdm(self.dictionary, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', lock_args=False, leave=False, position=pbar_loc):
            tot_match = 0
            max_match = -float('inf')
            for hints, repeats in zip(*np.unique(_generate_all_hints(guess, self.candidates), return_counts=True, axis=0)):
                num_match = self._get_match_indices(guess, hints).size
                tot_match += num_match * repeats
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


def process(progress_queue, initial_guess, initial_hints):
    pbar_loc = progress_queue.get() if progress_queue is not None else 1
    try:
        def recursive(args, d, level=0):
            temp_eng = Engine()
            for g,h in args:
                temp_eng.apply_guess(g,h)

            for guess in d:
                for hints in np.unique(_generate_all_hints(_encode(guess), temp_eng.candidates), axis=0):
                    if (hints==1).all():
                        continue
                    eng = Engine()
                    for g,h in args:
                        eng.apply_guess(g,h)
                    eng.apply_guess(guess, hints)
                    best = eng.get_best_guess(pbar_loc=pbar_loc)

                    k = tuple(hints.tolist())
                    d[guess][k] = {best: {}}
                    recursive(args + [(guess, hints)], d[guess][k], level+1)
            return d

        def init_decision_tree():
            initial_args = [(initial_guess, initial_hints)]

            eng = Engine()
            eng.apply_guess(*initial_args[0])
            best = eng.get_best_guess(pbar_loc=pbar_loc)

            initial_k = tuple(initial_hints.tolist())
            decision_tree = {initial_guess: {initial_k: {best: {}}}}
            return initial_args, decision_tree

        hint_key = tuple(initial_hints.tolist())
        return hint_key, recursive(*init_decision_tree())[initial_guess][hint_key]
    finally:
        if progress_queue is not None:
            progress_queue.put(pbar_loc)


def build_lookup_table():
    from multiprocessing import Pool, Manager
    from functools import partial

    NUM_WORKERS = 64
    STARTING_WORD = 'serai'

    hints = np.unique(_generate_all_hints(_encode(STARTING_WORD), Engine().candidates), axis=0)
    hints = hints[np.any(hints!=1,axis=1)]

    lookup_table = {STARTING_WORD: {}}

    with tqdm(total=len(hints), desc=f'Processing {STARTING_WORD} hints', ncols=90, position=0) as pbar:
        if NUM_WORKERS > 1:
            with Manager() as manager:
                progress_queue = manager.Queue()
                for i in range(NUM_WORKERS):
                    progress_queue.put(i+1)

                proc_fun = partial(process, progress_queue, STARTING_WORD)

                with Pool(initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),), processes=NUM_WORKERS) as pool:
                    for k, branch in tqdm(pool.imap_unordered(proc_fun, hints), desc=f'Processing {STARTING_WORD} hints'):
                        lookup_table[STARTING_WORD][k] = branch
                        pbar.update()
        else:
            for h in hints:
                k, branch = process(None, STARTING_WORD, h)
                lookup_table[STARTING_WORD][k] = branch
                pbar.update()

    with open('lookup_table_par.pkl', 'wb') as f:
        pkl.dump(lookup_table, f)


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
    from collections import defaultdict

    counts = defaultdict(list)
    for word in tqdm(map(_decode,Engine().candidates), ncols=0, total=len(Engine().candidates)):
        counts[count_guesses(word)].append(word)
    max_guesses = max(counts)
    total = 0
    s = 0
    for k in sorted(counts.keys()):
        if k > 6:
            print(f'{k}: ({len(counts[k])}) {counts[k]}')
        s += k * len(counts[k])
        total += len(counts[k])
    print('max guesses =',max(counts))
    print('avg guesses =',s / total)

    # interactive_helper()

    # build_lookup_table()
