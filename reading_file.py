
def read_file():
    fileName = 'names.txt'
    words = open(fileName, 'r').read().splitlines()
    chars = sorted(list(set(''.join(words))))
    stoi = {c: i+1 for i, c in enumerate(chars)}
    stoi['.'] = 0 
    itos = {i: c for c, i in stoi.items()}
    return words, stoi, itos