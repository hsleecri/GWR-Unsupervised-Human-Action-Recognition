# Explanations are added by Hyeonsu Lee
import numpy as np
import math
from heapq import nsmallest
from typing import Tuple, Union, Callable, Any
import matplotlib.pyplot as plt
import gtls 


class NoLabelGammaGWR:
    # Initialization
    def __init__(self):
        self.iterations=0

    # Computes the context coefficients(alpha_w) for the GWR algorithm.
    # Takes in the number of coefficients to compute and returns an array of computed coefficients.
    def compute_alphas(self, num_coeff) -> np.array:
        alpha_w = np.zeros(num_coeff)
        for h in range(0, len(alpha_w)):
            alpha_w[h] = np.exp(-h)
        alpha_w[:] = alpha_w[:] / sum(alpha_w)
        return alpha_w

    # Computes the Euclidean distance between two input vectors (x and y)
    def compute_distance(self, x, y) -> float:
        return np.linalg.norm(np.dot(self.alphas.T, (x - y)))

    # Finds the two best matching units(BMUs) from a list of distances.
    # Takes in a list of distances and returns indices of the two BMUs and their distances
    def find_bs(self, dis) -> Tuple[int, float, int]:
        bs = nsmallest(2, ((k, i) for i, k in enumerate(dis)))
        return bs[0][1], bs[0][0], bs[1][1]

    # Finds the best matching unit(BMU) for a given input vector.
    # Takes in the input vector and an optional argument 's_best' to indicate if the second BMU should also be returned
    # Returns the index of the BMU and its distance
    # or if 's_best' is True, a tuple of the indices of the two BMUs and their distances
    def find_bmus(self, input_vector, **kwargs) -> Union[Callable[[np.ndarray], Any], Tuple[int, float]]:
        second_best = kwargs.get('s_best', False)
        distances = np.zeros(self.num_nodes)
        for i in range(0, self.num_nodes):
            distances[i] = self.compute_distance(self.weights[i], input_vector)
        if second_best:
            # Compute best and second-best matching units
            return self.find_bs(distances)
        else:
            b_index = distances.argmin()
            b_distance = distances[b_index]
            return b_index, b_distance

    # Expands the input matrix by adding a row and column of zeros to be bottom right of the matrix
    def expand_matrix(self, matrix) -> np.array:
        ext_matrix = np.hstack((matrix, np.zeros((matrix.shape[0], 1))))
        ext_matrix = np.vstack((ext_matrix, np.zeros((1, ext_matrix.shape[1]))))
        return ext_matrix

    # Initializes the GWR network. Takes in a dataset(ds)
    # a flag indicating if the initial weights should be randomly assigned, and several optional arguments
    # Sets up the basic structure of the GWR network, including the number of nodes, dimensions, and context;
    # initializing weights and habituation counters; creating edge and age matrices;
    # and initializing the coefficients.
    def init_network(self, ds, random, **kwargs) -> None:

        assert self.iterations < 1, "Can't initialize a trained network"
        assert ds is not None, "Need a dataset to initialize a network"

        # Lock to prevent training
        self.locked = False

        # Start with 2 neurons
        self.num_nodes = 2
        self.dimension = ds.vectors.shape[1] #여기서 열 개수 뽑아오는 거 같은데 왜 안되지
        self.num_context = kwargs.get('num_context', 0)
        self.depth = self.num_context + 1
        empty_neuron = np.zeros((self.depth, self.dimension))
        self.weights = [empty_neuron, empty_neuron]

        # Global context
        self.g_context = np.zeros((self.depth, self.dimension))

        # Create habituation counters
        self.habn = [1, 1]

        # Create edge and age matrices
        self.edges = np.ones((self.num_nodes, self.num_nodes))
        self.ages = np.zeros((self.num_nodes, self.num_nodes))

        # Label histograms
        # empty_label_hist = -np.ones(ds.num_classes)
        # self.alabels = [empty_label_hist, empty_label_hist]

        # Initialize weights
        self.random = random
        if self.random:
            init_ind = np.random.randint(0, ds.vectors.shape[0], 2)
        else:
            init_ind = list(range(0, self.num_nodes))
        for i in range(0, len(init_ind)):
            self.weights[i] = ds.vectors[init_ind[i]]
            # self.alabels[i][int(ds.labels[i])] = 1
            print(self.weights[i])

        # Context coefficients
        self.alphas = self.compute_alphas(self.depth)

    # Adds a new node to the GWR network.
    # Takes in the index of the best matching unit and updates the weights and structure of the network accordingly.
    def add_node(self, b_index) -> None:
        new_neuron = np.array(np.dot(self.weights[b_index] + self.g_context, self.new_node))
        self.weights.append(new_neuron)
        self.num_nodes += 1

    # Updates the weight of a node in the GWR network.
    # Takes in the index of the node to be updated and the learning rate(epsilon)
    # and updates the node's weight based on the current context and habituation counter
    def update_weight(self, index, epsilon) -> None:
        delta = np.dot((self.g_context - self.weights[index]), (epsilon * self.habn[index]))
        self.weights[index] = self.weights[index] + delta

    # Decreases the habituation counter of a node in the GWR network.
    # Takes in the index of the node to be habituated and the decay constant(tau)
    # and updates the habituation counter accordingly.
    def habituate_node(self, index, tau, **kwargs) -> None:
        new_node = kwargs.get('new_node', False)
        if not new_node:
            self.habn[index] += tau * 1.05 * (1 - self.habn[index]) - tau
        else:
            self.habn.append(1)

    # Updates the weight of the neighboring nodes of the best matching unit
    def update_neighbors(self, index, epsilon) -> None:
        b_neighbors = np.nonzero(self.edges[index])
        for z in range(0, len(b_neighbors[0])):
            neIndex = b_neighbors[0][z]
            self.update_weight(neIndex, epsilon)
            self.habituate_node(neIndex, self.tau_n, new_node=False)

    # Updates the activation levels of the node's associated class labels.
    # def update_labels(self, bmu, label, **kwargs) -> None:
    #     new_node = kwargs.get('new_node', False)
    #     if not new_node:
    #         for a in range(0, self.num_classes):
    #             if a == label:
    #                 self.alabels[bmu][a] += self.a_inc
    #             else:
    #                 if label != -1:
    #                     self.alabels[bmu][a] -= self.a_dec
    #                     if (self.alabels[bmu][a] < 0):
    #                         self.alabels[bmu][a] = 0
    #     else:
    #         new_alabel = np.zeros(self.num_classes)
    #         if label != -1:
    #             new_alabel[int(label)] = self.a_inc
    #         self.alabels.append(new_alabel)

    # Updates the edges and ages of the connections between nodes
    def update_edges(self, fi, si, **kwargs) -> None:
        new_index = kwargs.get('new_index', False)
        self.ages += 1
        if not new_index:
            self.edges[fi, si] = 1
            self.edges[si, fi] = 1
            self.ages[fi, si] = 0
            self.ages[si, fi] = 0
        else:
            self.edges = self.expand_matrix(self.edges)
            self.ages = self.expand_matrix(self.ages)
            self.edges[fi, si] = 0
            self.edges[si, fi] = 0
            self.ages[fi, si] = 0
            self.ages[si, fi] = 0
            self.edges[fi, new_index] = 1
            self.edges[new_index, fi] = 1
            self.edges[si, new_index] = 1
            self.edges[new_index, si] = 1

    # Removes edges that are older than the specified maximum age
    def remove_old_edges(self) -> None:
        for i in range(0, self.num_nodes):
            neighbours = np.nonzero(self.edges[i])
            for j in neighbours[0]:
                if self.ages[i, j] > self.max_age:
                    self.edges[i, j] = 0
                    self.edges[j, i] = 0
                    self.ages[i, j] = 0
                    self.ages[j, i] = 0

    # Removes nodes that have no connections to other nodes
    def remove_isolated_nodes(self) -> None:
        if self.num_nodes > 2:
            ind_c = 0
            rem_c = 0
            while (ind_c < self.num_nodes):
                neighbours = np.nonzero(self.edges[ind_c])
                if len(neighbours[0]) < 1:
                    self.weights.pop(ind_c)
                    # self.alabels.pop(ind_c)
                    self.habn.pop(ind_c)
                    self.edges = np.delete(self.edges, ind_c, axis=0)
                    self.edges = np.delete(self.edges, ind_c, axis=1)
                    self.ages = np.delete(self.ages, ind_c, axis=0)
                    self.ages = np.delete(self.ages, ind_c, axis=1)
                    self.num_nodes -= 1
                    rem_c += 1
                else:
                    ind_c += 1
            print("(-- Removed %s neuron(s))" % rem_c)

    # Trains GammaGWR
    def train_gammagwr(self, ds, epochs, a_threshold, beta, l_rates, max_age) -> None:

        assert not self.locked, "Network is locked. Unlock to train."
        assert ds.vectors.shape[1] == self.dimension, "Wrong dimensionality"
        

        self.samples = ds.vectors.shape[0]
        self.max_epochs = epochs
        self.a_threshold = a_threshold
        self.epsilon_b, self.epsilon_n = l_rates
        self.beta = beta

        self.hab_threshold = 0.1
        self.tau_b = 0.3
        self.tau_n = 0.1
        self.max_nodes = self.samples  # OK for batch, bad for incremental
        self.max_neighbors = 6
        self.max_age = max_age
        self.new_node = 0.5
        # self.num_classes = ds.num_classes
        self.a_inc = 1
        self.a_dec = 0.1

        node_counts = []
        node_counts.append(self.num_nodes)
        epochs_list = []
        
        

        # Start training
        error_counter = np.zeros(self.max_epochs)
        previous_bmu = np.zeros((self.depth, self.dimension))

        for epoch in range(0, self.max_epochs):
            epochs_list.append(epoch)

            for iteration in range(0, self.samples):

                # Generate input sample
                self.g_context[0] = ds.vectors[iteration]
                # label = ds.labels[iteration]

                # Update global context
                for z in range(1, self.depth):
                    self.g_context[z] = (self.beta * previous_bmu[z]) + ((1 - self.beta) * previous_bmu[z - 1])

                # Find the best and second-best matching neurons
                b_index, b_distance, s_index = self.find_bmus(self.g_context, s_best=True)

                # Quantization error
                error_counter[epoch] += b_distance

                # Compute network activity
                a = math.exp(-b_distance)

                # Store BMU at time t for t+1
                previous_bmu = self.weights[b_index]

                if (a < self.a_threshold
                        and self.habn[b_index] < self.hab_threshold
                        and self.num_nodes < self.max_nodes):

                    # Add new neuron
                    n_index = self.num_nodes
                    self.add_node(b_index)

                    # Add label histogram
                    # self.update_labels(n_index, label, new_node=True)

                    # Update edges and ages
                    self.update_edges(b_index, s_index, new_index=n_index)

                    # Habituation counter
                    self.habituate_node(n_index, self.tau_b, new_node=True)

                else:
                    # Habituate BMU
                    self.habituate_node(b_index, self.tau_b)

                    # Update BMU's weight vector
                    self.update_weight(b_index, self.epsilon_b)

                    # Update BMU's edges // Remove BMU's oldest ones
                    self.update_edges(b_index, s_index)

                    # Update BMU's neighbors
                    self.update_neighbors(b_index, self.epsilon_n)

                    # Update BMU's label histogram
                    # self.update_labels(b_index, label)

                self.iterations += 1

            # Remove old edges
            self.remove_old_edges()

            # Average quantization error (AQE)
            error_counter[epoch] /= self.samples

            print("(Epoch: %s, NN: %s, ATQE: %s)" %
                  (epoch + 1, self.num_nodes, error_counter[epoch]))
            node_counts.append(self.num_nodes)

            # Remove isolated neurons
            self.remove_isolated_nodes()
            


        
        
        plt.plot(node_counts)
        plt.title('Number of nodes during training')
        plt.xlabel('Epoch')
        plt.ylabel('Number of nodes')
        plt.title("Number of Nodes vs Epochs")
        plt.yticks(np.arange(int(min(node_counts)), int(max(node_counts))+1, 1))
        plt.show()

        plt.plot(error_counter)
        plt.title('Number of nodes during training')
        plt.xlabel('Epoch')
        plt.ylabel('ATQE')
        plt.title("ATQE vs Epochs")
        plt.show()

    # def convert_to_array(my_net):
    #     nodes = my_net.nodes
    #     num_nodes = len(nodes)
    #     dists = np.zeros((num_nodes, num_nodes))
    #     for i in range(num_nodes):
    #         for j in range(i, num_nodes):
    #             dist = np.linalg.norm(nodes[i].position - nodes[j].position)
    #             dists[i, j] = dist
    #             dists[j, i] = dist
    #     return -dists ** 2 / (2 * np.median(dists))

    def convert_to_array(self):
        dists = np.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            for j in range(i,self.num_nodes):
                dist = np.linalg.norm(self.num_nodes[i].position - self.num_nodes[j].position)
                dists[i, j] = dist
                dists[j, i] = dist
        return -dists ** 2 / (2 * np.median(dists))

    # Tests GammaGWR
    def test_gammagwr(self, test_ds, **kwargs):
        test_accuracy = kwargs.get('test_accuracy', None)
        self.bmus_index = -np.ones(self.samples)
        self.bmus_label = -np.ones(self.samples)
        self.bmus_activation = np.zeros(self.samples)

        input_context = np.zeros((self.depth, self.dimension))

        if test_accuracy: acc_counter = 0

        for i in range(0, test_ds.vectors.shape[0]):
            input_context[0] = test_ds.vectors[i]

            # Find the BMU
            b_index, b_distance = self.find_bmus(input_context)
            self.bmus_index[i] = b_index
            self.bmus_activation[i] = math.exp(-b_distance)
            # self.bmus_label[i] = np.argmax(self.alabels[b_index])

            for j in range(1, self.depth):
                input_context[j] = input_context[j - 1]

            if test_accuracy:
                # if self.bmus_label[i] == test_ds.labels[i]:
                    acc_counter += 1

        if test_accuracy:
            self.test_accuracy = acc_counter / test_ds.vectors.shape[0]