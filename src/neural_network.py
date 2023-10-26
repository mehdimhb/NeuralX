import numpy as np
from numpy.typing import NDArray, ArrayLike
from typing import Callable
from function_utils import sigmoid, relu, sigmoid_backward, relu_backward
import h5py
import logging
from src.confusion_matrix import ConfusionMatrix

logging.basicConfig(
    filename='file.log', filemode='w', level=logging.DEBUG,
    format='%(message)s'
)


class Layer:
    def __init__(
        self,
        id: int,
        no_of_units: int,
        no_of_units_of_previous_layer: int,
        weight_initialization_method: str,
        activation_function: str,
        keeping_neuron_probability_in_dropout: float
    ) -> None:
        self.id = id
        self.next: Layer = None
        self.prev: Layer = None
        self.no_of_units = no_of_units
        self.W = self.weight_initialization(weight_initialization_method, no_of_units, no_of_units_of_previous_layer)
        self.dW: NDArray
        self.b = np.zeros((no_of_units, 1))
        self.db: NDArray
        self.Z: NDArray
        self.dZ: NDArray
        self.A: NDArray
        self.dA: NDArray
        self._dropout: NDArray
        self.keep_probability = keeping_neuron_probability_in_dropout
        self.act_fun_forward, self.act_fun_backward = self.function_detection(activation_function)

    def dropout(self, A: NDArray, forward: bool = False) -> NDArray:
        if forward:
            self._dropout = np.random.rand(self.A.shape[0], self.A.shape[1])
            self._dropout = (self._dropout < self.keep_probability).astype(int)
        A *= self._dropout
        A /= self.keep_probability
        return A

    def weight_initialization(
        self,
        weight_initialization_method: str,
        no_of_units: int,
        no_of_units_of_previous_layer: int = 0
    ):
        match weight_initialization_method:
            case 'zero':
                return np.zeros((no_of_units, no_of_units_of_previous_layer))
            case 'he':
                return np.random.randn(no_of_units, no_of_units_of_previous_layer)*np.sqrt(2/no_of_units_of_previous_layer)
            case 'xavier':
                return np.random.randn(no_of_units, no_of_units_of_previous_layer)*np.sqrt(2/(no_of_units+no_of_units_of_previous_layer))

    def function_detection(self, f: str) -> tuple[Callable[[NDArray], NDArray], Callable[[NDArray, NDArray], NDArray]]:
        match f:
            case 'sigmoid':
                return sigmoid, sigmoid_backward
            case 'relu':
                return relu, relu_backward

    def __repr__(self) -> str:
        return f"{self.id}:{self.no_of_units}:{self.W.shape}"


class Layers:
    def __init__(self):
        self.first: Layer = None
        self.last: Layer = None

    def __repr__(self) -> str:
        r = ""
        for layer in self:
            if layer is self.last:
                r += str(layer)
            else:
                r += str(layer) + " -> "
        return r

    def __len__(self):
        return np.sum([1 for _ in self])

    def __iter__(self):
        layer = self.first
        while layer:
            yield layer
            layer = layer.next

    def __reversed__(self):
        layer = self.last
        while layer:
            yield layer
            layer = layer.prev

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
        layers_description: list[tuple[int, None, None] | tuple[int, str, str]],
        training_set: NDArray = None,
        training_set_labels: NDArray = None,
        learning_rate: float = 0.0075,
    ) -> None:
        self.tX = training_set
        self.tY = training_set_labels
        self.learning_rate = learning_rate

        self.iteration = 0
        self.layers = self.build_layers(layers_description)
        self.costs = []

    def build_layers(self, layers_description: list[tuple[int, None, None] | tuple[int, str, str]]) -> Layers:
        layers = Layers()
        for layer in range(1, len(layers_description)):
            layers.add(Layer(
                layer+1,
                layers_description[layer][0],
                layers_description[layer-1][0],
                layers_description[layer][1],
                layers_description[layer][2],
                layers_description[layer][3]
            ))
        return layers

    def forward(self, data: NDArray, dropout: bool) -> None:
        for layer in self.layers:
            if layer is self.layers.first:
                layer.Z = np.dot(layer.W, data) + layer.b
            else:
                layer.Z = np.dot(layer.W, layer.prev.A) + layer.b
            layer.A = layer.act_fun_forward(layer.Z)
            if dropout:
                layer.A = layer.dropout(layer.A, forward=True)

    def backward(self, labels: NDArray, reg_lambd: float, dropout: bool) -> None:
        size = labels.shape[1]
        for layer in reversed(self.layers):
            if layer is self.layers.last:
                layer.dA = - (np.divide(labels, layer.A) - np.divide(1 - labels, 1 - layer.A))
            layer.dZ = layer.act_fun_backward(layer.dA, layer.Z)
            layer.db = np.sum(layer.dZ, axis=1, keepdims=True)/size
            if layer is self.layers.first:
                layer.dW = np.dot(layer.dZ, self.tX.T)/size + reg_lambd*layer.W/size
            else:
                layer.dW = np.dot(layer.dZ, layer.prev.A.T)/size + reg_lambd*layer.W/size
                layer.prev.dA = np.dot(layer.W.T, layer.dZ)
                if dropout:
                    layer.prev.dA = layer.prev.dropout(layer.prev.dA)

    def update_layers(self) -> None:
        for layer in self.layers:
            layer.W = layer.W - self.learning_rate * layer.dW
            layer.b = layer.b - self.learning_rate * layer.db

    def compute_cost(self, labels: NDArray, regularization: str = None, reg_lambd: float = 0) -> float:
        size = labels.shape[1]
        cost = -(np.dot(labels, np.log(self.layers.last.A).T) + np.dot(1-labels, np.log(1-self.layers.last.A).T))/size
        if regularization is not None:
            regularization_cost = 0
            for layer in self.layers:
                if regularization == 'L1':
                    regularization_cost += np.sum(np.abs(layer.W))
                elif regularization == 'L2':
                    regularization_cost += np.sum(np.square(layer.W))
            cost += regularization_cost*reg_lambd/(2*size)
        return np.squeeze(cost)

    def train(
        self,
        no_of_iterations,
        training_set: NDArray = None,
        training_set_labels: NDArray = None,
        regularization: str = None,
        reg_lambd: float = 0,
        dropout: bool = False
    ) -> None:
        for i in range(no_of_iterations):
            self.forward(self.tX if training_set is None else training_set, dropout)
            self.backward(self.tY if training_set_labels is None else training_set_labels, reg_lambd, dropout)
            self.update_layers()
            self.iteration += 1
            if self.iteration % 100 == 0 or i == no_of_iterations-1:
                cost = self.compute_cost(self.tY if training_set is None else training_set_labels, regularization, reg_lambd)
                logging.debug(f"Cost after iteration {self.iteration}: {cost}")
                self.costs.append(cost)

    def predict(self, data: NDArray):
        self.forward(data, False)
        return np.squeeze(self.layers.last.A)


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
    layer = [
        (12288, None, None, None),
        (7, 'he', 'relu', 0.8),
        (5, 'he', 'relu', 0.7),
        (1, 'he', 'sigmoid', 1)
    ]
    model = NeuralNetwork(layer, training_set, training_set_labels)
    model.train(2000, dropout=True)
    logging.debug(model.costs)
    cm = ConfusionMatrix(model, test_set, test_set_labels)
    logging.debug(cm)
    logging.debug(cm.statistics())
