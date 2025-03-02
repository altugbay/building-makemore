import torch
import torch.nn.functional as F
from draw import draw

SEED = 2147483647

fileName = 'names.txt'
words = open('names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {c: i+1 for i, c in enumerate(chars)}
stoi['.'] = 0 
itos = {i: c for c, i in stoi.items()}

def getProbabilityMatrix(isDraw, printFirstRow):
    N = torch.zeros(27, 27, dtype=torch.int32)
    for word in words:
        chs = ['.'] + list(word) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            N[ix1, ix2] += 1

    if isDraw:
        draw(N, itos) 


    if printFirstRow:
        print("first row counts", N[0])
        p = N[0].float()
        p = p / p.sum()
        print("first row probability", p)
        g = torch.Generator().manual_seed(SEED)
        selectedTensor = torch.multinomial(p, num_samples=1, replacement=True, generator=g)
        print("selectedFromFirstRow", selectedTensor)
        print("The selected character from the first row is", itos[selectedTensor.item()])    
        print("------------------------------------")

    P = (N+1).float() # +1 is for regularization
    P = P / P.sum(1, keepdim=True)
    return P, itos

def getNeuralNetwork(printLikelihood):

    xs, ys = [], []
    #for word in words[:1]:
    for word in words:    
        chs = ['.'] + list(word) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            xs.append(ix1)
            ys.append(ix2)
            
    xs = torch.tensor(xs)
    ys = torch.tensor(ys)
    numberOfExamples = xs.nelement()
    #print("number examples", numberOfExamples)

    # randomly initiate 27 neurons' weights. each neuron receives 27 inputs
    g = torch.Generator().manual_seed(SEED)
    W = torch.randn(27, 27, generator=g, requires_grad=True)

    # gradient descent
    for i in range(200):
        # forward pass
        xenc = F.one_hot(xs, num_classes=27).float() # input to network: one hot encoding of the character
        #print("xenc.shape", xenc.shape)
        logits = xenc @ W #predict log counts
        counts = logits.exp() # counts, equivalent to N
        probs = counts / counts.sum(1, keepdim=True) # probabilities for next character
        # last two lines called softmax
        # we need to get if ys = [5, 13, 13, 1, 0] from probs and it means probs[0, 5], probs[1, 13], probs[2, 13], probs[3, 1], probs[4, 0]
        #print(probs[torch.arange(numberOfExamples), ys])
        loss = -probs[torch.arange(numberOfExamples), ys].log().mean() + 0.1 * (W ** 2).mean() # + 0.1 * (W ** 2).mean() part is called regularization
        #print("loss", loss.item())
        # forward pass

        # backward pass
        W.grad = None
        loss.backward()
        W.data += -50 * W.grad
        # backward pass
    
    if printLikelihood:
        printLikelihood(xs, ys, itos, probs)

    return W, itos

def printLikelihood(xs, ys, itos, probs):
    nlls = torch.zeros(5)
    for i in range(5):
        x = xs[i].item()
        y = ys[i].item()
        print('-------')
        print(f'bigram example {i+1}: {itos[x]}{itos[y]} (index {x},{y})')
        print('input to neuron net:', x)
        print('output probabilities from the neuron net:', probs[i])
        print('label (actual next character):', y)
        p = probs[i, y]
        print('probability assigned by the net to the correct character:', p.item())
        logp = torch.log(p)
        print('log likelihood:', logp.item())  
        nll = -logp
        print('negative log likelihood:', nll.item())
        nlls[i] = nll

    print('=================')
    print('average negative log likelihood:', nlls.mean().item())
    print('=================')