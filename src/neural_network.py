import numpy as np
from numpy.typing import NDArray, ArrayLike
from function_utils import sigmoid, relu, sigmoid_backward, relu_backward
import h5py
import logging

logging.basicConfig(
    filename='file2.log', filemode='w', level=logging.DEBUG,
    format='%(asctime)s - %(message)s', datefmt='%H:%M:%S'
)


class Layer:
    def __init__(self, id: int, no_of_units: int, no_of_units_of_previous_layer: int, act_fun_forward, act_fun_backward) -> None:
        self.id = id
        self.next: Layer = None
        self.prev: Layer = None
        self.no_of_units = no_of_units
        self.W = np.random.randn(no_of_units, no_of_units_of_previous_layer)/np.sqrt(no_of_units_of_previous_layer)
        self.dW: NDArray
        self.b = np.zeros((no_of_units, 1))
        self.db: NDArray
        self.A: NDArray
        self.dA: NDArray
        self.Z: NDArray
        self.dZ: NDArray
        self.act_fun_forward = act_fun_forward
        self.act_fun_backward = act_fun_backward

    def __repr__(self) -> str:
        return f"{self.id}:{self.no_of_units}"


class Layers:
    def __init__(self):
        self.first: Layer = None
        self.last: Layer = None

    def __repr__(self) -> str:
        layer = self.first
        r = ""
        while layer:
            if layer is self.last:
                r += str(layer)
            else:
                r += str(layer) + " -> "
            layer = layer.next
        return r

    def add(self, new_layer: Layer) -> None:
        if self.first is None:
            self.first = new_layer
            self.last = new_layer
        else:
            self.last.next = new_layer
            new_layer.prev = self.last
            self.last = new_layer


class NeuralNetwork:
    def __init__(
        self,
        training_set: NDArray,
        training_set_labels: NDArray,
        layers_dimension: ArrayLike,
        learning_rate: float = 0.0075,
        threshold: float = 0.5
    ) -> None:
        self.tX = training_set
        self.tY = training_set_labels
        self.learning_rate = learning_rate
        self.threshold = threshold

        self.iteration = 0
        self.features, self.size_of_training_set = self.tX.shape
        self.layers = self.initialize_layers(layers_dimension)
        self.costs = []

    def initialize_layers(self, layers_dimension: ArrayLike) -> Layers:
        layers = Layers()
        for layer in range(len(layers_dimension)):
            if layer == 0:
                layers.add(Layer(layer+1, layers_dimension[layer], self.features, relu, relu_backward))
            elif layer == len(layers_dimension)-1:
                layers.add(Layer(layer+1, layers_dimension[layer], layers_dimension[layer-1], sigmoid, sigmoid_backward))
            else:
                layers.add(Layer(layer+1, layers_dimension[layer], layers_dimension[layer-1], relu, relu_backward))
        return layers

    def forward(self):
        layer = self.layers.first
        while layer:
            if layer is self.layers.first:
                layer.Z = np.dot(layer.W, self.tX) + layer.b
            else:
                layer.Z = np.dot(layer.W, layer.prev.A) + layer.b
            layer.A = layer.act_fun_forward(layer.Z)
            layer = layer.next

    def compute_cost(self):
        cost = -(np.dot(self.tY, np.log(self.layers.last.A).T) + np.dot(1-self.tY, np.log(1-self.layers.last.A).T))/self.size_of_training_set
        return np.squeeze(cost)

    def backward(self):
        layer = self.layers.last
        while layer:
            if layer is self.layers.last:
                layer.dA = - (np.divide(self.tY, layer.A) - np.divide(1 - self.tY, 1 - layer.A))
            layer.dZ = layer.act_fun_backward(layer.dA, layer.Z)
            layer.db = np.sum(layer.dZ, axis=1, keepdims=True)/self.size_of_training_set
            if layer is self.layers.first:
                layer.dW = np.dot(layer.dZ, self.tX.T)/self.size_of_training_set
            else:
                layer.dW = np.dot(layer.dZ, layer.prev.A.T)/self.size_of_training_set
                layer.prev.dA = np.dot(layer.W.T, layer.dZ)
            layer = layer.prev

    def update_layers(self):
        layer = self.layers.first
        while layer:
            layer.W = layer.W - self.learning_rate * layer.dW
            layer.b = layer.b - self.learning_rate * layer.db
            layer = layer.next

    def train(self, no_of_iterations):
        for i in range(no_of_iterations):
            self.forward()
            self.backward()
            self.update_layers()
            self.iteration += 1
            if self.iteration % 100 == 0 or i == no_of_iterations-1:
                cost = self.compute_cost()
                logging.debug(f"Cost after iteration {self.iteration}: {cost}")
                self.costs.append(cost)


def load_dataset():
    with h5py.File('data/train_catvnoncat.h5', 'r') as f:
        training_set = np.array(f["train_set_x"][:])
        training_set_labels = np.array(f["train_set_y"][:])

    with h5py.File('data/test_catvnoncat.h5', 'r') as f:
        test_set = np.array(f["test_set_x"][:])
        test_set_labels = np.array(f["test_set_y"][:])

    training_set = (training_set.reshape(training_set.shape[0], -1).T)/255
    training_set_labels = training_set_labels.reshape((1, training_set_labels.shape[0]))
    test_set = (test_set.reshape(test_set.shape[0], -1).T)/255
    test_set_labels = test_set_labels.reshape((1, test_set_labels.shape[0]))

    return training_set, training_set_labels, test_set, test_set_labels


if __name__ == "__main__":
    training_set, training_set_labels, test_set, test_set_labels = load_dataset()
    nn = NeuralNetwork(training_set, training_set_labels, [20, 7, 5, 1])
    nn.train(1000)
    # logging.debug(nn.costs)
    nn.train(1000)
