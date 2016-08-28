import numpy as np
import matplotlib.pyplot as plt


def create_plot_0(pos):
    plt.plot(pos[0, :], pos[1, :],
             linestyle='',
             marker='o', markeredgecolor='k',
             markerfacecolor=[1, 1, 1], markersize=18)
    plt.xlim((-1, 1))
    plt.ylim((-1, 1))
    plt.savefig('figure_0.png')
    plt.close()
    # plt.show()


def create_plot_1(pos):
    for idx_src in range(pos.shape[1]):
        for idx_dst in range(pos.shape[1]):
            plt.plot(pos[0, [idx_src, idx_dst]],
                     pos[1, [idx_src, idx_dst]],
                     color=[0, 0, 0],
                     marker='o', markeredgecolor='k',
                     markerfacecolor=[1, 1, 1], markersize=18)
    plt.xlim((-1, 1))
    plt.ylim((-1, 1))
    plt.savefig('figure_1.png')
    plt.close()
    # plt.show()


def create_plot_2(pos, syn):
    for idx_src in range(pos.shape[1]):
        for idx_dst in range(pos.shape[1]):
            if syn[idx_src, idx_dst] < 0:
                color = [0.8, 0.8, 0.8]
            else:
                color = [0.0, 0.0, 0.0]
            plt.plot(pos[0, [idx_src, idx_dst]],
                     pos[1, [idx_src, idx_dst]],
                     color=color,
                     marker='o', markeredgecolor='k',
                     markerfacecolor=[1, 1, 1], markersize=18)
    plt.xlim((-1, 1))
    plt.ylim((-1, 1))
    plt.savefig('figure_2.png')
    plt.close()
    # plt.show()


def create_plot_3(pos, syn):
    for idx_src in range(pos.shape[1]):
        for idx_dst in range(pos.shape[1]):
            if syn[idx_src, idx_dst] < 0:
                color = [0.8, 0.8, 0.8]
            else:
                color = [0.0, 0.0, 0.0]
            w = int(10 * abs(syn[idx_src, idx_dst])) + 1
            plt.plot(pos[0, [idx_src, idx_dst]],
                     pos[1, [idx_src, idx_dst]],
                     color=color, linewidth=w,
                     marker='o', markeredgecolor='k',
                     markerfacecolor=[1, 1, 1], markersize=18)
    plt.xlim((-1, 1))
    plt.ylim((-1, 1))
    plt.savefig('figure_3.png')
    plt.close()
    # plt.show()


def create_evo_map(val):
    plt.imshow(val, cmap=plt.cm.gray, aspect='auto', interpolation='nearest')
    plt.xlabel('Neuron')
    plt.ylabel('Time Step')
    plt.savefig('figure_4.png')
    plt.close()
    # plt.show()
    

def update(val, syn, step):
    for idx_neu in range(val.shape[1]):
        val[step, idx_neu] = np.sum(syn[:, idx_neu] * val[step-1, :])
    val[step, :] = np.clip(val[step, :], 0, 1)
    return val
    

if __name__ == '__main__':
    num_neurons = 10
    neuron_values = np.zeros((50, num_neurons))
    neuron_values[0, :] = np.random.random(
        size=(1, neuron_values.shape[1]))

    neuron_positions = np.zeros((2, num_neurons))
    angle = 0.
    angle_update = 2 * np.pi / num_neurons
    for i in range(num_neurons):
        neuron_positions[:, i] = (np.sin(angle), np.cos(angle))
        angle += angle_update

    create_plot_0(neuron_positions)
    
    synapses = np.random.random((num_neurons, num_neurons))
    # Rescale to [-1, 1]
    synapses = synapses * 2 - 1

    create_plot_1(neuron_positions)
    create_plot_2(neuron_positions, synapses)
    create_plot_3(neuron_positions, synapses)

    for step in range(1, 49):
        neuron_values = update(neuron_values, synapses, step)

    create_evo_map(neuron_values)
