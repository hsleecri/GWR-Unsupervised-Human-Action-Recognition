import csv
import pickle
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
import pandas as pd


def import_network(file_name, NetworkClass):
    """ Import pickled network from file
    """
    file = open(file_name, 'br')
    data_pickle = file.read()
    file.close()
    net = NetworkClass()
    net.__dict__ = pickle.loads(data_pickle)
    return net


def export_network(file_name, net) -> None:
    """ Export pickled network to file
    """
    file = open(file_name, 'wb')
    file.write(pickle.dumps(net.__dict__))
    file.close()


def export_result(file_name, net, ds) -> None:
    net.bmus_index = -np.ones(net.samples)
    net.bmus_activation = np.zeros(net.samples)
    for i in range(0, net.samples):
        input = ds.vectors[i]
        b_index, b_distance = net.find_bmus(input)
        net.bmus_index[i] = b_index
        net.bmus_activation[i] = math.exp(-b_distance)
    

    #print(net.bmus_index)
    print("len(net.weights)=")
    print(len(net.weights))
    for ni in range(len(net.weights)):
        print(net.weights[ni][0])
    print("ends")
    gwr_node = [x[0] for x in net.weights]
    df_node = pd.DataFrame(gwr_node)

    weights_file_name= str(file_name+"weight.csv")
    df_node.to_csv(path_or_buf=weights_file_name, index=True)
    df = pd.DataFrame(net.bmus_index)
    #print(df)
    df.to_csv(path_or_buf=file_name, index=False, header=False)
    plt.figure(figsize=(20, 10))
    plt.xlabel('Number of frame')
    plt.ylabel('Number of cluster')
    sns.scatterplot(net.bmus_index)
    plt.show()

# cnt = []
# for i in range(len(net.clusters)):
#     cnt.append([i,len(net.clusters[i])])

# df_cnt = pd.DataFrame(cnt,columns=['cluster이름','개수'])

# file = open(file_name, 'wb')
# file.write(pickle.dumps(net.__dict__))
# file.close()

def export_epi_result(file_name, net, ds) -> None:
    net.bmus_index = -np.ones(ds.vectors.shape[0])
    net.bmus_activation = np.zeros(ds.vectors.shape[0])
    #net.temporal_ind = np.zeros(ds.vectors.shape[0])
    print(net.temporal)
    for i in range(0, ds.vectors.shape[0]):
        input = ds.vectors[i]
        b_index, b_distance = net.find_bmus(input)
        net.bmus_index[i] = b_index
        net.bmus_activation[i] = math.exp(-b_distance)
        #net.temporal_ind[i] = net.temporal[-1,i]
    

    #print(net.bmus_index)
    #net.bmus_index 를 csv로 export해주는 코드
    df = pd.DataFrame(net.bmus_index)
    #df = pd.concat([df, pd.DataFrame(net.temporal_ind[i])], axis=1)
    #print(df)
    df.to_csv(path_or_buf=file_name, index=False, header=False)
    plt.figure(figsize=(20, 10))
    plt.xlabel('Number of frame')
    plt.ylabel('Number of cluster')
    sns.scatterplot(net.bmus_index)
    plt.show()


def load_file(file_name) -> np.ndarray:
    """ Load dataset from file
    """
    # df = pd.read_csv(file_name)
    # df = df.dropna(axis=0)
    # df.to_csv(file_name)
    reader = csv.reader(open(file_name, "r", encoding="UTF8"), delimiter=',')
    x_rdr = list(reader)
    return np.array(x_rdr).astype('float')


def normalize_data(data) -> np.ndarray:
    """ Normalize data vectors
    """
    for i in range(0, data.shape[1]):
        max_col = max(data[:, i])
        min_col = min(data[:, i])
        for j in range(0, data.shape[0]):
            data[j, i] = (data[j, i] - min_col) / (max_col - min_col)
    return data


def plot_network(net, edges, labels) -> None:
    """ 2D plot
    """
    # Plot network
    # This just plots the first two dimensions of the weight vectors.
    # For better visualization, PCA over weight vectors must be performed.
    #print(net.weights)
    ccc = ['black', 'blue', 'red', 'green', 'yellow',
           'cyan', 'magenta', '0.75', '0.15', '1']
    plt.figure()
    dim_net = True if len(net.weights[0].shape) < 2 else False
    for ni in range(len(net.weights)):
        
        if labels:
            plindex = np.argmax(net.alabels[ni])
            if dim_net:
                plt.scatter(net.weights[ni][2], net.weights[ni]
                            [3], color=ccc[plindex], alpha=.5)
            else:
                plt.scatter(net.weights[ni][0, 2], net.weights[ni]
                            [0, 3], color=ccc[plindex], alpha=.5)
        else:
            if dim_net:
                plt.scatter(net.weights[ni][2], net.weights[ni][3], alpha=.5)
            else:
                plt.scatter(net.weights[ni][0, 2],
                            net.weights[ni][0, 3], alpha=.5)
        if edges:
            for nj in range(len(net.weights)):
                if net.edges[ni, nj] > 0:
                    if dim_net:
                        plt.plot([net.weights[ni][2], net.weights[nj][2]],
                                 [net.weights[ni][3], net.weights[nj][3]],
                                 'gray', alpha=.3)
                    else:
                        plt.plot([net.weights[ni][0, 2], net.weights[nj][0, 2]],
                                 [net.weights[ni][0, 3], net.weights[nj][0, 3]],
                                 'gray', alpha=.3)
    plt.show()


def plot_gamma_network(net, edges, labels) -> None:
    """ 2D plot
    """
    # Plot network
    # This just plots the first two dimensions of the weight vectors.
    # For better visualization, PCA over weight vectors must be performed.
    #print(net.weights)
    ccc = ['black', 'blue', 'red', 'green', 'yellow',
           'cyan', 'magenta', '0.75', '0.15', '1']
    plt.figure()
    dim_net = True if len(net.weights[0].shape) < 2 else False
    for ni in range(len(net.weights)):
        
        if labels:
            plindex = np.argmax(net.alabels[ni])
            if dim_net:
                plt.scatter(net.weights[ni][2], net.weights[ni]
                            [3], color=ccc[plindex], alpha=.5)
            else:
                plt.scatter(net.weights[ni][0, 2], net.weights[ni]
                            [0, 3], color=ccc[plindex], alpha=.5)
        else:
            if dim_net:
                plt.scatter(net.weights[ni][2], net.weights[ni][3], alpha=.5)
            else:
                plt.scatter(net.weights[ni][0, 2],
                            net.weights[ni][0, 3], alpha=.5)
        if edges:
            for nj in range(len(net.weights)):
                if net.edges[ni, nj] > 0:
                    if dim_net:
                        plt.plot([net.weights[ni][2], net.weights[nj][2]],
                                 [net.weights[ni][3], net.weights[nj][3]],
                                 'gray', alpha=.3)
                    else:
                        plt.plot([net.weights[ni][0, 2], net.weights[nj][0, 2]],
                                 [net.weights[ni][0, 3], net.weights[nj][0, 3]],
                                 'gray', alpha=.3)
    plt.show()


def bk_frame_plot_network(weights, alabels, edges, edge, labels, iterations) -> None:
    """ 2D plot
    """
    # Plot network
    # This just plots the first two dimensions of the weight vectors.
    # For better visualization, PCA over weight vectors must be performed.
    ccc = ['black', 'blue', 'red', 'green', 'yellow',
           'cyan', 'magenta', '0.75', '0.15', '1']
    plt.figure()
    plt.title("iterations = ", iterations)
    plt.xlim([0, 1])      # X축의 범위: [xmin, xmax]
    plt.ylim([0, 1])     # Y축의 범위: [ymin, ymax]
    dim_net = True if len(weights[0].shape) < 2 else False
    for ni in range(len(weights)):
        
        if labels:
            plindex = np.argmax(alabels[ni])
            if dim_net:
                plt.scatter(weights[ni][1], weights[ni][2],
                            color=ccc[plindex], alpha=.5)
            else:
                plt.scatter(weights[ni][0, 1], weights[ni]
                            [0, 2], color=ccc[plindex], alpha=.5)
        else:
            if dim_net:
                plt.scatter(weights[ni][1], weights[ni][2], alpha=.5)
            else:
                plt.scatter(weights[ni][0, 1], weights[ni][0, 2], alpha=.5)
        if edge:
            for nj in range(len(weights)):
                if edges[ni, nj] > 0:
                    if dim_net:
                        plt.plot([weights[ni][1], weights[nj][1]],
                                 [weights[ni][2], weights[nj][2]],
                                 'gray', alpha=.3)
                    else:
                        plt.plot([weights[ni][0, 1], weights[nj][0, 1]],
                                 [weights[ni][0, 2], weights[nj][0, 2]],
                                 'gray', alpha=.3)
    plt.show()


def bk_no_label_frame_plot_network_learning(weights, edges, edge, black, iterations, NN, learn_add_nodes, learn_update_BMU, learn_neighbors, learn_remove_edges, learn_remove_nodeWeights, learn_new_edges, AQE) -> None:
    learn_add_nodes = sorted(list(set(learn_add_nodes)))
    learn_update_BMU = sorted(list(set(learn_update_BMU)))
    learn_neighbors = sorted(list(set(learn_neighbors)))
    print('add', learn_add_nodes)
    print('bmu', learn_update_BMU)
    print('naver', learn_neighbors)
    print('len_weights', len(weights))

    """ 2D plot
    """
    # Plot network
    # This just plots the first two dimensions of the weight vectors.
    # For better visualization, PCA over weight vectors must be performed.
    ccc = ['black', 'blue', 'red', 'green', 'yellow',
           'cyan', 'magenta', '0.75', '0.15', '1']
    plt.figure()
    plt.title("(iterations: %s, NN: %s, AQE : %s)" %
              (iterations, NN, AQE))
    plt.xlim([0, 1])      # X축의 범위: [xmin, xmax]
    plt.ylim([0, 1])     # Y축의 범위: [ymin, ymax]
    dim_net = True if len(weights[0].shape) < 2 else False

    for ni in range(len(learn_remove_nodeWeights)):
        if dim_net:
            plt.scatter(
                learn_remove_nodeWeights[ni][1][0], learn_remove_nodeWeights[ni][1][1], marker='x', c='black', alpha=.5)
        else:
            plt.scatter(
                learn_remove_nodeWeights[ni][1][0], learn_remove_nodeWeights[ni][1][1], marker='x', c='black', alpha=.5)

    for ni in range(len(weights)):
        nodecolor = 'gray'
        # 기본노드 : darkred,기본 엣지: black ,새로생성 : dodgerblue, BMU: red, 학습된이웃노드:tomato, 삭제노드: black, 삭제엣지:grey

        if ni in learn_neighbors:
            nodecolor = 'yellow'
        if ni in learn_update_BMU:
            nodecolor = 'red'
        if ni in learn_add_nodes:
            nodecolor = 'dodgerblue'
        if black:
            if dim_net:
                plt.scatter(weights[ni][0], weights[ni]
                            [1], c=nodecolor, alpha=1)
            else:
                plt.scatter(weights[ni][0, 0], weights[ni]
                            [0, 1], c=nodecolor, alpha=1)
        else:
            if dim_net:
                plt.scatter(weights[ni][0], weights[ni]
                            [1], c=nodecolor, alpha=1)
            else:
                plt.scatter(weights[ni][0, 0], weights[ni]
                            [0, 1], c=nodecolor, alpha=1)

        if edge:
            for nj in range(len(weights)):
                if edges[ni, nj] > 0:
                    if dim_net:
                        plt.plot([weights[ni][0], weights[nj][0]],
                                 [weights[ni][1], weights[nj][1]],
                                 'black', alpha=.5)
                    else:
                        plt.plot([weights[ni][0, 0], weights[nj][0, 0]],
                                 [weights[ni][0, 1], weights[nj][0, 1]],
                                 'black', alpha=.5)

    for ni in range(len(learn_remove_edges)):
        remove_edge = learn_remove_edges[ni]
        if dim_net:
            plt.plot([weights[remove_edge[0]][0], weights[remove_edge[1]][0]],
                     [weights[remove_edge[0]][1], weights[remove_edge[1]][1]],
                     'gray', linestyle=(0, (1, 5)), alpha=1)
        else:
            plt.plot([weights[remove_edge[0]][0], weights[remove_edge[1]][0]],
                     [weights[remove_edge[0]][1], weights[remove_edge[1]][1]],
                     'gray', linestyle=(0, (1, 5)), alpha=1)

    for ni in range(len(learn_new_edges)):
        new_edge = learn_new_edges[ni]
        if dim_net:
            plt.plot([weights[new_edge[0]][0], weights[new_edge[1]][0]],
                     [weights[new_edge[0]][1], weights[new_edge[1]][1]],
                     'orangered', alpha=0.7)
        else:
            plt.plot([weights[new_edge[0]][0], weights[new_edge[1]][0]],
                     [weights[new_edge[0]][1], weights[new_edge[1]][1]],
                     'orangered', alpha=0.7)

    plt.show()


def bk_no_label_frame_plot_network(weights, edges, edge, black, iterations, NN) -> None:
    """ 2D plot
    """
    # Plot network
    # This just plots the first two dimensions of the weight vectors.
    # For better visualization, PCA over weight vectors must be performed.
    ccc = ['black', 'blue', 'red', 'green', 'yellow',
           'cyan', 'magenta', '0.75', '0.15', '1']
    plt.figure()
    plt.title("(iterations: %s, NN: %s)" %
              (iterations, NN))
    plt.xlim([0, 1])      # X축의 범위: [xmin, xmax]
    plt.ylim([0, 1])     # Y축의 범위: [ymin, ymax]
    dim_net = True if len(weights[0].shape) < 2 else False
    for ni in range(len(weights)):
        if black:
            if dim_net:
                plt.scatter(weights[ni][0], weights[ni]
                            [1], c='black', alpha=.5)
            else:
                plt.scatter(weights[ni][0, 0], weights[ni]
                            [0, 1], c='black', alpha=.5)
        else:
            if dim_net:
                plt.scatter(weights[ni][0], weights[ni][1], alpha=.5)
            else:
                plt.scatter(weights[ni][0, 0], weights[ni][0, 1], alpha=.5)

        if edge:
            for nj in range(len(weights)):
                if edges[ni, nj] > 0:
                    if dim_net:
                        plt.plot([weights[ni][0], weights[nj][0]],
                                 [weights[ni][1], weights[nj][1]],
                                 'gray', alpha=.3)
                    else:
                        plt.plot([weights[ni][0, 0], weights[nj][0, 0]],
                                 [weights[ni][0, 1], weights[nj][0, 1]],
                                 'gray', alpha=.3)
    plt.show()


class IrisDataset:
    """ Create an instance of Iris dataset
    """

    def __init__(self, file, normalize):
        self.name = 'IRIS'
        self.file = file
        self.normalize = normalize
        self.num_classes = 3

        raw_data = load_file(self.file)

        self.labels = raw_data[:, raw_data.shape[1]-1]
        self.vectors = raw_data[:, 0:raw_data.shape[1]-1]

        label_list = list()
        for label in self.labels:
            if label not in label_list:
                label_list.append(label)
        n_classes = len(label_list)

        assert self.num_classes == n_classes, "Inconsistent number of classes"

        if self.normalize:
            self.vectors = normalize_data(self.vectors)


class bk_label_Dataset:
    """ Create an instance of any label dataset
    """

    def __init__(self, file, normalize, num_classes):
        self.name = 'data_set'
        self.file = file
        self.normalize = normalize
        self.num_classes = num_classes

        raw_data = load_file(self.file)
        

        self.labels = raw_data[:, raw_data.shape[1]-1]
        self.vectors = raw_data[:, 0:raw_data.shape[1]-1]

        label_list = list()
        for label in self.labels:
            if label not in label_list:
                label_list.append(label)
        n_classes = len(label_list)

        assert self.num_classes == n_classes, "Inconsistent number of classes"

        if self.normalize:
            self.vectors = normalize_data(self.vectors)

class bk_no_label_Dataset:
    """ for dataset without any label in it
    """

    def __init__(self, file, normalize):
        self.name = 'no_label'
        self.file = file
        self.normalize = normalize

        raw_data = load_file(self.file)
        self.vectors = raw_data

        if self.normalize:
            self.vectors = normalize_data(self.vectors)