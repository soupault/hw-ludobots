from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt


def MatrixCreate(rows, columns):
    return np.zeros((rows, columns))


def MatrixRandomize(v):
    return np.random.random(size=v.shape)


def MatrixPerturb(p, prob):
    out = p.copy()
    probs = np.random.random(size=p.shape)
    for i in range(out.size):
        if probs[0, i] <= prob:
            out[0, i] = np.random.random()
    return out


def Fitness(v):
    return np.mean(v)


def PlotVectorAsLine(fits):
    plt.plot(range(fits.size), fits.T)
    plt.xlim((-10, 5000))
    plt.ylim((0, 1))
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    

if __name__ == '__main__':
    Genes = MatrixCreate(50, 5000)
    
    for trial in range(1):
        parent = MatrixCreate(1, 50)
        parent = MatrixRandomize(parent)
        parentFitness = Fitness(parent)
        fits = MatrixCreate(1, 5000)
        
        for currentGeneration in range(5000):
            print(currentGeneration, parentFitness)
            child = MatrixPerturb(parent, 0.05)
            childFitness = Fitness(child)
            if childFitness > parentFitness:
                parent = child
                parentFitness = childFitness
            fits[0, currentGeneration] = parentFitness

            Genes[:, currentGeneration] = parent[0, :]
            
        # PlotVectorAsLine(fits)
    plt.imshow(Genes, cmap=plt.cm.gray, aspect='auto', interpolation='nearest')
    plt.xlabel('Generation')
    plt.ylabel('Gene')
    plt.show()
