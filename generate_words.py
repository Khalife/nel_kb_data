import pdb
import numpy as np


def edits0(word):
    "All edits that are one edit away from `word`."
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]

    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    #pdb.set_trace()
    replaces = [L + c + R[1:]  for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    
    random_letter = letters[np.random.randint(len(letters))]
    random_delete = np.random.randint(len(deletes))
    random_transpose = np.random.randint(len(transposes))
    random_replace = np.random.randint(len(replaces))
    random_insert = np.random.randint(len(inserts))

    deletes = deletes[random_delete]
    transposes = transposes[random_transpose]
    replaces = replaces[random_replace]
    inserts = inserts[random_insert]

    last_index = np.random.randint(4)

    return [deletes, transposes, replaces, inserts][last_index]

def edits1(word):
    "All edits that are one edit away from `word`."
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    #pdb.set_trace()
    replaces = [L + c + R[1:]  for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word): 
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))


def edits3(words_list):
    main_words_list = words_list[:300]
    n = len(main_words_list)
    if n <= 10:
        return words_list
    random_indexes = np.random.randint(0, n, 10) 
    return [main_words_list[ri] for ri in random_indexes]

     
if __name__ == "__main__":
    result = edits0("united states")
    results = edits3("The United States of America commonly known as the United States or America, is a country comprising 50 states".split()) 
    pdb.set_trace()
