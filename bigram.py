import torch
import torch.nn.functional as F
from constants import SEED
from draw import drawProbabilityMatrix
from reading_file import read_file
from multinomial_example import multinomialExample


# GOAL: maximize the likelihood of the data
# equivalent it means we want to maximize log likelihood
# it means we want to minimize negative log likelihood
# equivalent it means we want to minimize average log likelihood
# a * b * c (where a and b and c are the probabilities) = log(a) + log(b) + log(c)
words, stoi, itos = read_file()

def getBigramProbabilityMatrix(printLikelihood, isDraw, printFirstRow):
    N = torch.zeros(27, 27, dtype=torch.int32)
    for word in words:
        chs = ['.'] + list(word) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            N[ix1, ix2] += 1

    if isDraw:
        drawProbabilityMatrix(N, itos) 


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

    if printLikelihood: 
         printNegativeLikelihoodFromProbabilityMatrix(P)

    return P, itos

def printNegativeLikelihoodFromProbabilityMatrix(P):
    loglikelihood = 0
    n=0
    for word in words:
            chs = ['.'] + list(word) + ['.']
            for ch1, ch2 in zip(chs, chs[1:]):
                ix1 = stoi[ch1]
                ix2 = stoi[ch2]
                prob = P[ix1, ix2]
                logprob = torch.log(prob)
                loglikelihood += logprob
                n += 1
                #print(f'{ch1}{ch2}: {prob:.4f} (log likelihood: {logprob:.4f})')
    
    print('=================')        
    print('average negative log likelihood:', -loglikelihood / n)
    print('=================')   

def getBigramNeuralNetwork(printLikelihood):

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
        printNegativeLikelihoodFromNN(xs, ys, itos, probs)

    return W, itos

def printNegativeLikelihoodFromNN(xs, ys, itos, probs):
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

def samplingFromModel(P, W): 
    # generate 10 random names
    g = torch.Generator().manual_seed(SEED)
    for _ in range(10):
        out = []
        ix = 0
        while True:
            if not neural_network:
                p = P[ix]
            else:
                xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
                logits = xenc @ W
                counts = logits.exp() # counts, equivalent to N
                p = counts / counts.sum(1, keepdim=True) # probabilities for next character

            # finding index with multinomial distribution
            ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
            out.append(itos[ix])
            if ix == 0:
                break
        print(''.join(out))

neural_network = True
#multinomialExample() # this will print the probability of each element in the tensor, the sampling of the multinomial distribution and a line to separate the outputs
P, W = None, None
if not neural_network:
    P, itos = getBigramProbabilityMatrix(printLikelihood=False, isDraw=False, printFirstRow=False)
else:
    W, itos = getBigramNeuralNetwork(printLikelihood=False)

samplingFromModel(P, W)        