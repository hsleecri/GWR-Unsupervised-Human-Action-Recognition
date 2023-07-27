"""
gwr-tb :: Associative GWR based on Marsland et al. (2002)'s Grow-When-Required network
@last-modified: 25 January 2019
@author: German I. Parisi (german.parisi@gmail.com)

"""
import gtls
import scipy.spatial
import math
import numpy as np
from heapq import nsmallest
from typing import Tuple, Union, Callable, Any


class AssociativeGWR:

    def __init__(self):
        self.iterations = 0

    def compute_distance(self, x, y, m) -> float:
        return np.linalg.norm(x-y) if m else scipy.spatial.distance.cosine(x, y)

    def find_bs(self, dis) -> Tuple[int, float, int]:
        bs = nsmallest(2, ((k, i) for i, k in enumerate(dis)))
        return bs[0][1], bs[0][0], bs[1][1]

    def find_bmus(self, input_vector, **kwargs) -> Union[Callable[[np.ndarray], Any], Tuple[int, float]]:
        second_best = kwargs.get('s_best', False)
        distances = np.zeros(self.num_nodes)
        for i in range(0, self.num_nodes):
            distances[i] = self.compute_distance(
                self.weights[i], input_vector, self.dis_metric)

        if second_best:
            # Compute the best and second-best matching units
            return self.find_bs(distances)
        else:
            b_index = distances.argmin()
            b_distance = distances[b_index]
            return b_index, b_distance

    def expand_matrix(self, matrix) -> np.array:
        ext_matrix = np.hstack((matrix, np.zeros((matrix.shape[0], 1))))
        ext_matrix = np.vstack(
            (ext_matrix, np.zeros((1, ext_matrix.shape[1]))))
        return ext_matrix

    def init_network(self, ds, random) -> None:

        assert self.iterations < 1, "Can't initialize a trained network"
        assert ds is not None, "Need a dataset to initialize a network"

        # Lock to prevent training
        self.locked = False

        # Start with 2 neurons with dimensionality given by dataset
        self.num_nodes = 2
        self.dimension = ds.vectors.shape[1]
        empty_weight = np.zeros(self.dimension)
        self.weights = [empty_weight, empty_weight]

        # Create habituation counters
        self.habn = [1, 1]

        # Create edge and age matrices
        self.edges = np.ones((self.num_nodes, self.num_nodes))
        self.ages = np.zeros((self.num_nodes, self.num_nodes))

        # Label histograms
        empty_label_hist = -np.ones(ds.num_classes)
        self.alabels = [empty_label_hist, empty_label_hist]

        # 학습 시각화를 위한 중간 노드, 엣지 저장
        self.learn_add_nodes = []
        self.learn_update_BMU = []
        self.learn_remove_nodeWeights = []
        self.learn_neighbors = []
        self.learn_remove_edges = []
        self.learn_new_edges = []
        self.learn_update_input = []

        # Initialize weights
        self.random = random
        if self.random:
            init_ind = np.random.randint(0, ds.vectors.shape[0], 2)
        else:
            init_ind = list(range(0, self.num_nodes))
        for i in range(0, len(init_ind)):
            self.weights[i] = ds.vectors[init_ind[i]]
            self.alabels[i][int(ds.labels[i])] = 1
            print(self.weights[i])

    def bk_no_lable_init_network(self, ds, random) -> None:

        assert self.iterations < 1, "Can't initialize a trained network"
        assert ds is not None, "Need a dataset to initialize a network"

        # Lock to prevent training
        self.locked = False

        # Start with 2 neurons with dimensionality given by dataset
        self.num_nodes = 2
        self.dimension = ds.vectors.shape[1]
        empty_weight = np.zeros(self.dimension)
        self.weights = [empty_weight, empty_weight]

        # Create habituation counters
        self.habn = [1, 1]

        # Create edge and age matrices
        self.edges = np.ones((self.num_nodes, self.num_nodes))
        self.ages = np.zeros((self.num_nodes, self.num_nodes))

        # 학습 시각화를 위한 중간 노드, 엣지 저장
        self.learn_add_nodes = []
        self.learn_update_BMU = []
        self.learn_neighbors = []
        self.learn_remove_nodeWeights = []
        self.learn_remove_edges = []
        self.learn_new_edges = []
        self.learn_update_input = []

        # Initialize weights
        self.random = random
        if self.random:
            init_ind = np.random.randint(0, ds.vectors.shape[0], 2)
        else:
            init_ind = list(range(0, self.num_nodes))
        for i in range(0, len(init_ind)):
            self.weights[i] = ds.vectors[init_ind[i]]
            print(self.weights[i])

    def add_node(self, b_index, input_vector) -> None:
        new_weight = np.array(
            np.dot(self.weights[b_index] + input_vector, self.new_node))
        self.weights.append(new_weight)
        self.num_nodes += 1
        self.learn_add_nodes.append(self.num_nodes-1)


    def update_weight(self, input, index, epsilon) -> None:
        delta = np.dot(
            (input - self.weights[index]), (epsilon * self.habn[index]))
        self.weights[index] = self.weights[index] + delta

    def habituate_node(self, index, tau, **kwargs) -> None:
        new_node = kwargs.get('new_node', False)
        if not new_node:
            self.habn[index] += tau * 1.05 * (1 - self.habn[index]) - tau
        else:
            self.habn.append(1)

    def update_neighbors(self, input, index, epsilon) -> None:
        b_neighbors = np.nonzero(self.edges[index])
        for z in range(0, len(b_neighbors[0])):
            neIndex = b_neighbors[0][z]
            self.update_weight(input, neIndex, epsilon)
            self.habituate_node(neIndex, self.tau_n, new_node=False)
            self.learn_neighbors.append(neIndex)

    def update_labels(self, bmu, label, **kwargs) -> None:
        new_node = kwargs.get('new_node', False)
        if not new_node:
            for a in range(0, self.num_classes):
                if a == label:
                    self.alabels[bmu][a] += self.a_inc
                else:
                    if label != -1:
                        self.alabels[bmu][a] -= self.a_dec
                        if (self.alabels[bmu][a] < 0):
                            self.alabels[bmu][a] = 0
        else:
            new_alabel = np.zeros(self.num_classes)
            if label != -1:
                new_alabel[int(label)] = self.a_inc
            self.alabels.append(new_alabel)

    def update_edges(self, fi, si, **kwargs) -> None:
        new_index = kwargs.get('new_index', False)
        self.ages += 1
        if not new_index:
            if (self.edges[fi, si] == 0):
                new_edge = [[fi, si], [si, fi]]
                self.learn_new_edges.extend(new_edge)
                self.learn_remove_edges = [
                item for item in self.learn_remove_edges if item not in new_edge]
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
            new_edge = [[fi, new_index], [new_index, fi],
                           [new_index, si], [si, new_index]]
            self.learn_new_edges.extend(new_edge)
            remove_edge = [[fi,si],[si,fi]]
            self.learn_new_edges = [
                item for item in self.learn_new_edges if item not in remove_edge]
            self.learn_remove_edges.extend(remove_edge)
            self.learn_remove_edges = [
                item for item in self.learn_remove_edges if item not in new_edge]

    def remove_old_edges(self) -> None:
        for i in range(0, self.num_nodes):
            neighbours = np.nonzero(self.edges[i])
            for j in neighbours[0]:
                if self.ages[i, j] > self.max_age:
                    self.edges[i, j] = 0
                    self.edges[j, i] = 0
                    self.ages[i, j] = 0
                    self.ages[j, i] = 0
                    old_edge = ([i, j], [j, i])
                    self.learn_remove_edges.extend(old_edge)
                    self.learn_new_edges = [
                        item for item in self.learn_new_edges if item not in old_edge]
                    # for item in self.learn_update_edges:
                    #     if item == [i,j] :
                    #         self.learn_update_edges.remove([i,j])
                    #     if item == [j,i] :
                    #         self.learn_update_edges.remove([j,i])

    def remove_isolated_nodes(self) -> None:
        ind_c = 0
        rem_c = 0
        while (ind_c < self.num_nodes):
            neighbours = np.nonzero(self.edges[ind_c])
            if len(neighbours[0]) < 1:
                self.learn_remove_nodeWeights.append([ind_c,self.weights[ind_c]])
                self.weights.pop(ind_c)
                self.alabels.pop(ind_c)
                self.habn.pop(ind_c)
                self.edges = np.delete(self.edges, ind_c, axis=0)
                self.edges = np.delete(self.edges, ind_c, axis=1)
                self.ages = np.delete(self.ages, ind_c, axis=0)
                self.ages = np.delete(self.ages, ind_c, axis=1)
                delNode = []
                for i in range(self.num_nodes):
                    delNode.append([ind_c, i])
                    delNode.append([i, ind_c])
                self.num_nodes -= 1
                rem_c += 1
                self.learn_add_nodes = [
                    item for item in self.learn_add_nodes if item != ind_c]
                self.learn_add_nodes = list(map(lambda x: x-1 if x >= ind_c else x,self.learn_add_nodes))
                self.learn_update_BMU = [
                    item for item in self.learn_update_BMU if item != ind_c]
                self.learn_update_BMU = list(map(lambda x: x-1 if x >= ind_c else x,self.learn_update_BMU))
                self.learn_neighbors = [
                    item for item in self.learn_neighbors if item != ind_c]
                self.learn_neighbors = list(map(lambda x: x-1 if x >= ind_c else x,self.learn_neighbors))
                self.learn_new_edges = list(map(lambda x: [x[0]-1,x[1]-1]  if x[0] >= ind_c and x[1] >= ind_c 
                    else ([x[0]-1,x[1]] if x[0] >= ind_c and x[1] < ind_c  
                    else ([x[0],x[1]-1] if x[0] < ind_c and x[1] >= ind_c else x) ),self.learn_new_edges))
                self.learn_remove_edges = list(map(lambda x: [x[0]-1,x[1]-1]  if x[0] >= ind_c and x[1] >= ind_c 
                    else ([x[0]-1,x[1]] if x[0] >= ind_c and x[1] < ind_c  
                    else ([x[0],x[1]-1] if x[0] < ind_c and x[1] >= ind_c else x) ),self.learn_remove_edges))

            else:
                ind_c += 1
        print("(-- Removed %s neuron(s))" % rem_c)

    def bk_no_label_remove_isolated_nodes(self) -> None:
        ind_c = 0
        rem_c = 0
        while (ind_c < self.num_nodes):
            neighbours = np.nonzero(self.edges[ind_c])
            if len(neighbours[0]) < 1:
                self.learn_remove_nodeWeights.append([ind_c,self.weights[ind_c]])
                self.weights.pop(ind_c)
                self.habn.pop(ind_c)
                self.edges = np.delete(self.edges, ind_c, axis=0)
                self.edges = np.delete(self.edges, ind_c, axis=1)
                self.ages = np.delete(self.ages, ind_c, axis=0)
                self.ages = np.delete(self.ages, ind_c, axis=1)
                delNode = []
                for i in range(self.num_nodes):
                    delNode.append([ind_c, i])
                    delNode.append([i, ind_c])
                self.num_nodes -= 1
                rem_c += 1
                self.learn_add_nodes = [
                    item for item in self.learn_add_nodes if item != ind_c]
                self.learn_add_nodes = list(map(lambda x: x-1 if x >= ind_c else x,self.learn_add_nodes))
                self.learn_update_BMU = [
                    item for item in self.learn_update_BMU if item != ind_c]
                self.learn_update_BMU = list(map(lambda x: x-1 if x >= ind_c else x,self.learn_update_BMU))
                self.learn_neighbors = [
                    item for item in self.learn_neighbors if item != ind_c]
                self.learn_neighbors = list(map(lambda x: x-1 if x >= ind_c else x,self.learn_neighbors))
                self.learn_new_edges = list(map(lambda x: [x[0]-1,x[1]-1]  if x[0] >= ind_c and x[1] >= ind_c 
                    else ([x[0]-1,x[1]] if x[0] >= ind_c and x[1] < ind_c  
                    else ([x[0],x[1]-1] if x[0] < ind_c and x[1] >= ind_c else x) ),self.learn_new_edges))
                self.learn_remove_edges = list(map(lambda x: [x[0]-1,x[1]-1]  if x[0] >= ind_c and x[1] >= ind_c 
                    else ([x[0]-1,x[1]] if x[0] >= ind_c and x[1] < ind_c  
                    else ([x[0],x[1]-1] if x[0] < ind_c and x[1] >= ind_c else x) ),self.learn_remove_edges))

            else:
                ind_c += 1
        print("(-- Removed %s neuron(s))" % rem_c)

    def bk_visualization_frame(self, ds, epochs, a_threshold, l_rates, max_age, frame) -> None:
        assert not self.locked, "Network is locked. Unlock to train."
        assert ds.vectors.shape[1] == self.dimension, "Wrong data dimensionality"

        self.samples = ds.vectors.shape[0]
        self.max_epochs = epochs
        self.a_threshold = a_threshold
        self.epsilon_b, self.epsilon_n = l_rates

        self.hab_threshold = 0.1
        self.tau_b = 0.3
        self.tau_n = 0.1
        self.max_nodes = self.samples  # OK for batch, bad for incremental
        self.dis_metric = 1  # 1 = Euclidean, 0 = Cosine
        self.max_neighbors = 6
        self.max_age = max_age
        self.new_node = 0.5
        self.num_classes = ds.num_classes
        self.a_inc = 1
        self.a_dec = 0.1

        # Start training
        error_counter = np.zeros(self.max_epochs)

        for epoch in range(0, self.max_epochs):

            for iteration in range(0, self.samples):

                # Generate input sample
                input = ds.vectors[iteration]
                label = ds.labels[iteration]

                # Find best and second-best matching neurons
                b_index, b_distance, s_index = self.find_bmus(
                    input, s_best=True)
                self.learn_update_BMU.append(b_index)

                # Quantization error
                error_counter[epoch] += b_distance

                # Compute network activity
                a = math.exp(-b_distance)

                if (a < self.a_threshold
                    and self.habn[b_index] < self.hab_threshold
                        and self.num_nodes < self.max_nodes):

                    # Add new neuron
                    n_index = self.num_nodes
                    self.add_node(b_index, input)

                    # Add label histogram
                    self.update_labels(n_index, label, new_node=True)

                    # Update edges and ages
                    self.update_edges(b_index, s_index, new_index=n_index)

                    # Habituation counter
                    self.habituate_node(n_index, self.tau_b, new_node=True)

                else:
                    # Habituate BMU
                    self.habituate_node(b_index, self.tau_b)

                    # Update BMU's weight vector
                    self.update_weight(input, b_index, self.epsilon_b)

                    # Update BMU's edges and ages
                    self.update_edges(b_index, s_index)

                    # Update BMU's neighbors
                    self.update_neighbors(input, b_index, self.epsilon_n)

                    # Update BMU's label histogram
                    self.update_labels(b_index, label)

                self.iterations += 1
                # Remove old edges
                self.remove_old_edges()
                # Remove isolated neurons
                self.remove_isolated_nodes()
                print("(iterations: %s, NN: %s)" %
                      (self.iterations, self.num_nodes))
                if self.iterations % frame == 0:
                    gtls.bk_frame_plot_network(
                        self.weights, self.alabels, self.edges, edge=True, labels=True, iterations=self.iterations)

            # Average quantization error (AQE)
            error_counter[epoch] /= self.samples

            print("(Epoch: %s, NN: %s, AQE: %s)" %
                  (epoch + 1, self.num_nodes, error_counter[epoch]))

    def bk_no_label_train(self, ds, epochs, a_threshold, l_rates, max_age, frame, black) -> None:
        assert not self.locked, "Network is locked. Unlock to train."
        assert ds.vectors.shape[1] == self.dimension, "Wrong data dimensionality"

        self.samples = ds.vectors.shape[0]
        self.max_epochs = epochs
        self.a_threshold = a_threshold
        self.epsilon_b, self.epsilon_n = l_rates

        self.hab_threshold = 0.1
        self.tau_b = 0.3
        self.tau_n = 0.1
        self.max_nodes = self.samples  # OK for batch, bad for incremental
        self.dis_metric = 1  # 1 = Euclidean, 0 = Cosine
        self.max_neighbors = 6
        self.max_age = max_age
        self.new_node = 0.5
        self.a_inc = 1
        self.a_dec = 0.1

        # Start training
        error_counter = np.zeros(self.max_epochs)

        for epoch in range(0, self.max_epochs):

            for iteration in range(0, self.samples):

                # Generate input sample
                input = ds.vectors[iteration]

                # Find best and second-best matching neurons
                b_index, b_distance, s_index = self.find_bmus(
                    input, s_best=True)
                self.learn_update_BMU.append(b_index)

                # Quantization error
                error_counter[epoch] += b_distance

                # Compute network activity
                a = math.exp(-b_distance)

                if (a < self.a_threshold
                    and self.habn[b_index] < self.hab_threshold
                        and self.num_nodes < self.max_nodes):

                    # Add new neuron
                    n_index = self.num_nodes
                    self.add_node(b_index, input)

                    # Update edges and ages
                    self.update_edges(b_index, s_index, new_index=n_index)

                    # Habituation counter
                    self.habituate_node(n_index, self.tau_b, new_node=True)

                else:
                    # Habituate BMU
                    self.habituate_node(b_index, self.tau_b)

                    # Update BMU's weight vector
                    self.update_weight(input, b_index, self.epsilon_b)

                    # Update BMU's edges and ages
                    self.update_edges(b_index, s_index)

                    # Update BMU's neighbors
                    self.update_neighbors(input, b_index, self.epsilon_n)

                self.iterations += 1

                # Remove old edges
                self.remove_old_edges()

                # Remove isolated neurons
                self.bk_no_label_remove_isolated_nodes()

                print("(iterations: %s, NN: %s)" %
                      (self.iterations, self.num_nodes))

                if self.iterations % frame == 0:
                    now_AQE = error_counter[epoch]/iteration
                    print("(Epoch: %s, NN: %s, AQE: %s)" %
                         (epoch + 1, self.num_nodes, error_counter[epoch]/iteration))
                    #gtls.bk_no_label_frame_plot_network(weights=self.weights, edges=self.edges, edge=True, black=black,iterations=self.iterations, NN=self.num_nodes)
                    gtls.bk_no_label_frame_plot_network_learning(weights=self.weights, edges=self.edges, edge=True, black=black,iterations=self.iterations, NN=self.num_nodes, 
                    learn_add_nodes=self.learn_add_nodes,learn_update_BMU = self.learn_update_BMU,learn_neighbors=self.learn_neighbors,learn_remove_edges=self.learn_remove_edges,learn_remove_nodeWeights=self.learn_remove_nodeWeights,learn_new_edges=self.learn_new_edges,
                    AQE=now_AQE)
                    # 학습 시각화를 위한 중간 노드, 엣지 초기화
                    self.learn_add_nodes.clear()
                    self.learn_update_BMU.clear()
                    self.learn_neighbors.clear()
                    self.learn_remove_edges.clear()
                    self.learn_remove_nodeWeights.clear()
                    self.learn_new_edges.clear()
                    self.learn_update_input.clear()



            # Average quantization error (AQE)
            error_counter[epoch] /= self.samples

            print("(Epoch: %s, NN: %s, AQE: %s)" %
                  (epoch + 1, self.num_nodes, error_counter[epoch]))

    def train_agwr(self, ds, epochs, a_threshold, l_rates, max_age) -> None:

        assert not self.locked, "Network is locked. Unlock to train."
        assert ds.vectors.shape[1] == self.dimension, "Wrong data dimensionality"

        self.samples = ds.vectors.shape[0]
        self.max_epochs = epochs
        self.a_threshold = a_threshold
        self.epsilon_b, self.epsilon_n = l_rates

        self.hab_threshold = 0.1
        self.tau_b = 0.3
        self.tau_n = 0.1
        self.max_nodes = self.samples  # OK for batch, bad for incremental
        self.dis_metric = 1  # 1 = Euclidean, 0 = Cosine
        self.max_neighbors = 6
        self.max_age = max_age
        self.new_node = 0.5
        self.num_classes = ds.num_classes
        self.a_inc = 1
        self.a_dec = 0.1

        # Start training
        error_counter = np.zeros(self.max_epochs)

        for epoch in range(0, self.max_epochs):

            for iteration in range(0, self.samples):

                # Generate input sample
                input = ds.vectors[iteration]
                label = ds.labels[iteration]

                # Find best and second-best matching neurons
                b_index, b_distance, s_index = self.find_bmus(
                    input, s_best=True)

                # Quantization error
                error_counter[epoch] += b_distance

                # Compute network activity
                a = math.exp(-b_distance)

                if (a < self.a_threshold
                    and self.habn[b_index] < self.hab_threshold
                        and self.num_nodes < self.max_nodes):

                    # Add new neuron
                    n_index = self.num_nodes
                    self.add_node(b_index, input)

                    # Add label histogram
                    self.update_labels(n_index, label, new_node=True)

                    # Update edges and ages
                    self.update_edges(b_index, s_index, new_index=n_index)

                    # Habituation counter
                    self.habituate_node(n_index, self.tau_b, new_node=True)

                else:
                    # Habituate BMU
                    self.habituate_node(b_index, self.tau_b)

                    # Update BMU's weight vector
                    self.update_weight(input, b_index, self.epsilon_b)

                    # Update BMU's edges and ages
                    self.update_edges(b_index, s_index)

                    # Update BMU's neighbors
                    self.update_neighbors(input, b_index, self.epsilon_n)

                    # Update BMU's label histogram
                    self.update_labels(b_index, label)

                self.iterations += 1

            # Remove old edges
            self.remove_old_edges()

            # Average quantization error (AQE)
            error_counter[epoch] /= self.samples

            print("(Epoch: %s, NN: %s, AQE: %s)" %
                  (epoch + 1, self.num_nodes, error_counter[epoch]))

        # Remove isolated neurons
        self.remove_isolated_nodes()

    def test_agwr(self, test_ds) -> None:
        self.bmus_index = -np.ones(self.samples)
        self.bmus_label = -np.ones(self.samples)
        self.bmus_activation = np.zeros(self.samples)
        acc_counter = 0
        for i in range(0, test_ds.vectors.shape[0]):
            input = test_ds.vectors[i]
            b_index, b_distance = self.find_bmus(input)
            self.bmus_index[i] = b_index
            self.bmus_activation[i] = math.exp(-b_distance)
            self.bmus_label[i] = np.argmax(self.alabels[b_index])

            if self.bmus_label[i] == test_ds.labels[i]:
                acc_counter += 1

        self.test_accuracy = acc_counter / test_ds.vectors.shape[0]
