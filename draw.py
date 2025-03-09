import matplotlib.pyplot as plt


def drawProbabilityMatrix(N, itos):
    plt.figure(figsize=(16, 16))
    plt.imshow(N, cmap='Blues')
    for i in range(27):
        for j in range(27):
            chstr = itos[i] + itos[j]
            plt.text(j, i, chstr, ha='center', va='bottom', color='gray')
            plt.text(j, i, N[i, j].item(), ha='center', va='top', color='gray')        

    plt.axis('off')
    plt.show()  

def drawEmbedingMatrix(C, itos):
    plt.figure(figsize=(8,8))
    plt.scatter(C[:,0].data, C[:,1].data, s=200)
    for i in range(C.shape[0]):
        plt.text(C[i,0].item(), C[i,1].item(), itos[i], ha="center", va="center", color='white')
    plt.grid('minor')
    plt.show()    