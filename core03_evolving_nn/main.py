from copy import (copy, deepcopy)

import numpy as np
import matplotlib.pyplot as plt


class ANN(object):
    def __init__(self, num_neurons, history_len=10):
        self.num_neurons = num_neurons
        self.synapses = np.zeros((self.num_neurons,)*2)
        self.history_neurons = np.zeros((history_len, num_neurons))

    def init_neurons(self, values):
        self.history_neurons = np.zeros_like(self.history_neurons)
        self.history_neurons[-1, :] = values

    def update_neurons(self, num_steps=1):
        for idx_step in range(num_steps):
            tmp_neurons = np.empty(self.num_neurons)
            for idx_neu in range(self.num_neurons):
                tmp_neurons[idx_neu] = \
                    np.sum(self.synapses[:, idx_neu] * self.get_neurons())
            tmp_neurons = np.clip(tmp_neurons, 0, 1)
            self.history_neurons = np.roll(self.history_neurons,
                                           axis=0, shift=-1)
            self.history_neurons[-1, :] = tmp_neurons

    def get_neurons(self):
        return self.history_neurons[-1, :]

    def randomize_synapses(self, limits=(0, 1)):
        if limits == (0, 1):
            self.synapses = np.random.random(self.synapses.shape)
        elif limits == (-1, 1):
            self.synapses = np.random.random(self.synapses.shape) * 2 - 1
        else:
            raise ValueError('Unsupported limits')

    def perturb_synapses(self, prob=0.05, limits=(0, 1)):
        probs = np.random.random(size=self.synapses.shape)
        mask = np.where(probs < prob)

        if limits == (0, 1):
            updates = np.random.random(self.synapses.shape)
        elif limits == (-1, 1):
            updates = np.random.random(self.synapses.shape) * 2 - 1
        else:
            raise ValueError('Unsupported limits')

        self.synapses[mask] = updates[mask]


def fitness_1(actual, desired):
    distance = np.sum((actual - desired)**2) / actual.size
    return 1 - distance


def fitness_2(neurons_mat):
    res = np.sum(np.abs(np.diff(neurons_mat[1:-2, :], axis=0))) + \
          np.sum(np.abs(np.diff(neurons_mat[1:-2, :], axis=1)))
    res /= 2*8*9
    return res


def create_evo_map(val, num_fig=0):
    plt.imshow(val, cmap=plt.cm.gray,
               aspect='auto', interpolation='nearest')
    plt.xlabel('Neuron')
    plt.ylabel('Update Step')
    plt.savefig('figure_{}.png'.format(num_fig))
    plt.close()
    # plt.show()


def create_fitness_plot(val, num_fig=0):
    plt.plot(history_fitness)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.xlim((0, 1000))
    plt.ylim((0, 1))
    plt.savefig('figure_{}.png'.format(num_fig))
    plt.close()
    # plt.show()


if __name__ == '__main__':
    parent = ANN(num_neurons=10)
    parent.randomize_synapses(limits=(-1, 1))
    parent.init_neurons(values=0.5)
    parent.update_neurons(num_steps=9)

    create_evo_map(parent.history_neurons, num_fig=0)

    desired_neurons = np.zeros_like(parent.get_neurons())
    desired_neurons[1::2] = 1

    fitness_parent = fitness_1(parent.get_neurons(), desired_neurons)
    # fitness_parent = fitness_2(parent.history_neurons)

    history_fitness = []
    history_fitness.append(fitness_parent)

    for idx_gen in range(1000):
        child = deepcopy(parent)
        child.init_neurons(values=0.5)
        child.perturb_synapses(limits=(-1, 1))
        child.update_neurons(num_steps=9)
        fitness_child = fitness_1(child.get_neurons(), desired_neurons)
        # fitness_child = fitness_2(child.history_neurons)

        if fitness_child > fitness_parent:
            parent.synapses = copy(child.synapses)
            fitness_parent = fitness_child

        history_fitness.append(fitness_parent)

    create_fitness_plot(history_fitness, num_fig=2)

    parent.init_neurons(values=0.5)
    parent.update_neurons(num_steps=9)
    
    create_evo_map(parent.history_neurons, num_fig=1)
    print(parent.get_neurons())
