import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

from constants import SEED
from reading_file import read_file # for making figures

words, stoi, itos = read_file()
trackStats = True
drawEmbedingMatrix = False

# build the dataset
block_size = 3 # context length: how many characters do we take to predict the next one?

def build_dataset(words):
  X, Y = [], []
  for w in words:
    context = [0] * block_size
    for ch in w + '.':
      ix = stoi[ch]
      X.append(context)
      Y.append(ix)
      #print(''.join(itos[i] for i in context), '--->', itos[ix])
      context = context[1:] + [ix] # crop and append

  X = torch.tensor(X)
  Y = torch.tensor(Y)
  print("XShape", X.shape, "YShape", Y.shape)
  return X, Y

random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr, Ytr = build_dataset(words[:n1]) # 80% of the data
Xdev, Ydev = build_dataset(words[n1:n2]) # 10% of the data
Xte, Yte = build_dataset(words[n2:]) # 10% of the data


n_emb = 10 # the dimension of the character embedding vectors
n_hidden = 200 # the number of neurons in the hidden layer of the MLP
vocab_size = len(stoi)

g = torch.Generator().manual_seed(SEED) # for reproducibility
C = torch.randn((vocab_size, n_emb), generator=g)
W1 = torch.randn((n_emb * block_size, n_hidden), generator=g) * (5/3)/((n_emb * block_size)**0.5) # * 0.2 fixing for too saturated tanh function
#b1 = torch.randn(n_hidden, generator=g) * 0.01 # fixing for too saturated tanh function
W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.01 # multiplying with a small number to have small initial weights
b2 = torch.randn(vocab_size, generator=g) * 0.0 # set to zero in order to have the same initial distribution as the logits


# BatchNorm parameters
bngain = torch.ones((1, n_hidden))
bnbias = torch.zeros((1, n_hidden))
bnmean_running = torch.zeros((1, n_hidden))
bnstd_running = torch.ones((1, n_hidden))

parameters = [C, W1, W2, b2, bngain, bnbias]

print("parameters", sum(p.nelement() for p in parameters)) # number of parameters in total
for p in parameters:
  p.requires_grad = True


max_steps = 200000
batch_size = 32
lossi = []

for i in range(max_steps):

    # minibatch construction
    ix = torch.randint(0, Xtr.shape[0], (batch_size,))
    Xb, Yb = Xtr[ix], Ytr[ix] # batch X,Y


    # forward pass
    emb = C[Xb] # embed the characters into vectors
    embcat = emb.view(emb.shape[0], -1) # concatenate the vectors
    # Linear Layer
    hpreact = embcat @ W1 #+ b1 # hidden layer pre-activation

    # BatchNorm layer
    # -------------------------------------------------------------
    bnmeani = hpreact.mean(0, keepdim=True)
    bnstdi = hpreact.std(0, keepdim=True)
    hpreact = bngain * (hpreact - bnmeani) / bnstdi + bnbias
    with torch.no_grad():
      bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
      bnstd_running = 0.999 * bnstd_running + 0.001 * bnstdi
    # -------------------------------------------------------------

    # Non-linearity
    h = torch.tanh(hpreact) # hidden layer
    logits = h @ W2 + b2 # output layer
    loss = F.cross_entropy(logits, Yb) # loss function
    
    
    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()      

    # update
    lr = 0.1 if i < 100000 else 0.01
    for p in parameters:
        p.data += -lr * p.grad
    
    # track stats
    if trackStats: 
      # track stats
      if i % 10000 == 0: # print every once in a while
        print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
      lossi.append(loss.log10().item())

if trackStats:
    plt.plot(lossi)
    plt.show()   
 

@torch.no_grad() # this decorator disables gradient tracking
def printLoss(label, X, Y):
    emb = C[X] # (N, block_size, n_emb) --- (32, 3, 2)
    embcat = emb.view(emb.shape[0], -1) # concat into (N, block_size * n_emb) --- (32, 6)
    hpreact = embcat @ W1 #+ b1
    hpreact = bngain * (hpreact - bnmean_running) / bnstd_running + bnbias
    h = torch.tanh(hpreact) # (N, n_hidden) --- (32, 100)
    logits = h @ W2 + b2 # (N, vocab_size) --- (32, 27)
    loss = F.cross_entropy(logits, Y)
    print(label, loss.item())

printLoss("Training loss", Xtr, Ytr)
printLoss("Validation loss", Xdev, Ydev)
printLoss("Test loss", Xte, Yte)


# sample from the model
g = torch.Generator().manual_seed(SEED+10)
for _ in range(20):
    out = []
    context = [0] * block_size # initialize with all ...
    while True:
      emb = C[torch.tensor([context])] # (1,block_size,d)
      h = torch.tanh(emb.view(1, -1) @ W1) # + b1)
      logits = h @ W2 + b2
      probs = F.softmax(logits, dim=1)
      ix = torch.multinomial(probs, num_samples=1, generator=g).item()
      context = context[1:] + [ix]
      out.append(ix)
      if ix == 0:
        break
    
    print(''.join(itos[i] for i in out))


# visualize dimensions 0 and 1 of the embedding matrix C for all characters
if drawEmbedingMatrix:
    drawEmbedingMatrix(C, itos)


