import torch
from multinomial_example import multinomialExample
from read_file import SEED, getProbabilityMatrix, getNeuralNetwork
import torch.nn.functional as F

neural_network = True
#multinomialExample() # this will print the probability of each element in the tensor, the sampling of the multinomial distribution and a line to separate the outputs

if not neural_network:
    P, itos = getProbabilityMatrix(draw=False, printFirstRow=False)
else:
    W, itos = getNeuralNetwork(printLikelihood=False)

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

    
    
 


