import torch

def multinomialExample() :
    g = torch.Generator().manual_seed(2147483647)
    p = torch.rand(3, generator=g)
    print(p/p.sum()) # it is the probability of each element in the tensor, so %60 for the first element (0), %30 for the second(1) and %10 for the third(2)
    print(torch.multinomial(p, 100, replacement=True, generator=g)) # this is the sampling of the multinomial distribution, it will return the index of the element that was sampled
    print("------------------------------------")
