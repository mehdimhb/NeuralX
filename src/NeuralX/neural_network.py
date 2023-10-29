import numpy as np
from numpy.typing import NDArray
from neuralx.function_utils import sigmoid, relu, sigmoid_backward, relu_backward


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
        self.vW = np.zeros(self.W.shape)
        self.vb = np.zeros(self.b.shape)
        self.sW = np.zeros(self.W.shape)
        self.sb = np.zeros(self.b.shape)
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
                return np.random.randn(no_of_units, no_of_units_of_previous_layer)\
                    * np.sqrt(2/(no_of_units+no_of_units_of_previous_layer))

    def function_detection(self, f: str):
        match f:
            case 'sigmoid':
                return sigmoid, sigmoid_backward
            case 'relu':
                return relu, relu_backward

    def __repr__(self) -> str:
        return f"Layer{self.id}({self.no_of_units})"


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
        layers_description: list,
        training_set: NDArray = None,
        training_set_labels: NDArray = None,
    ) -> None:
        self.training_set = training_set
        self.training_set_labels = training_set_labels
        self.epoch = 0
        self.layers = self.build_layers(layers_description)
        self.costs = {}

    def build_layers(self, layers_description: list) -> Layers:
        layers = Layers()
        for layer in range(1, len(layers_description)):
            layers.add(Layer(
                layer,
                layers_description[layer][0],
                layers_description[layer-1][0],
                layers_description[layer][1],
                layers_description[layer][2],
                layers_description[layer][3]
            ))
        return layers

    def forward(self, data: NDArray, dropout: bool, epsilon_mask: NDArray = None) -> None:
        for layer in self.layers:
            if layer is self.layers.first:
                if epsilon_mask is None:
                    layer.Z = np.dot(layer.W, data) + layer.b
                else:
                    layer.Z = np.dot(layer.W + epsilon_mask[layer][0], data) + layer.b + epsilon_mask[layer][1]
            else:
                if epsilon_mask is None:
                    layer.Z = np.dot(layer.W, layer.prev.A) + layer.b
                else:
                    layer.Z = np.dot(layer.W + epsilon_mask[layer][0], layer.prev.A) + layer.b + epsilon_mask[layer][1]
            layer.A = layer.act_fun_forward(layer.Z)
            if dropout:
                layer.A = layer.dropout(layer.A, forward=True)

    def backward(self, data: NDArray, labels: NDArray, reg_lambd: float, dropout: bool) -> None:
        size = labels.shape[1]
        for layer in reversed(self.layers):
            if layer is self.layers.last:
                layer.dA = - (np.divide(labels, layer.A) - np.divide(1 - labels, 1 - layer.A))
            layer.dZ = layer.act_fun_backward(layer.dA, layer.Z)
            layer.db = np.sum(layer.dZ, axis=1, keepdims=True)/size
            if layer is self.layers.first:
                layer.dW = np.dot(layer.dZ, data.T)/size + reg_lambd*layer.W/size
            else:
                layer.dW = np.dot(layer.dZ, layer.prev.A.T)/size + reg_lambd*layer.W/size
                layer.prev.dA = np.dot(layer.W.T, layer.dZ)
                if dropout:
                    layer.prev.dA = layer.prev.dropout(layer.prev.dA)

    def update_layers(self, optimization: dict, learning_rate: float) -> None:
        match optimization['name']:
            case 'gradient descent':
                for layer in self.layers:
                    layer.W = layer.W - learning_rate * layer.dW
                    layer.b = layer.b - learning_rate * layer.db
            case 'momentum':
                beta = optimization['beta']
                for layer in self.layers:
                    layer.vW = beta*layer.vW + (1-beta)*layer.dW
                    layer.vb = beta*layer.vb + (1-beta)*layer.db
                    layer.W = layer.W - learning_rate * layer.vW
                    layer.b = layer.b - learning_rate * layer.vb
            case 'adam':
                t = optimization['t']
                beta1 = optimization['beta1']
                beta2 = optimization['beta2']
                epsilon = optimization['epsilon']
                for layer in self.layers:
                    layer.vW = beta1*layer.vW + (1-beta1)*layer.dW
                    layer.vb = beta1*layer.vb + (1-beta1)*layer.db
                    vW_corrected = layer.vW/(1-np.power(beta1, t))
                    vb_corrected = layer.vb/(1-np.power(beta1, t))
                    layer.sW = beta2*layer.sW + (1-beta2)*np.power(layer.dW, 2)
                    layer.sb = beta2*layer.sb + (1-beta2)*np.power(layer.db, 2)
                    sW_corrected = layer.sW/(1-np.power(beta2, t))
                    sb_corrected = layer.sb/(1-np.power(beta2, t))
                    layer.W = layer.W - learning_rate * vW_corrected/(np.sqrt(sW_corrected)+epsilon)
                    layer.b = layer.b - learning_rate * vb_corrected/(np.sqrt(sb_corrected)+epsilon)

    def compute_cost(self, labels: NDArray, regularization: str, reg_lambd: float) -> float:
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

    def gradient_check(self, data: NDArray = None, labels: NDArray = None, epsilon: float = 1e-7) -> str:
        data = self.training_set if data is None else data
        labels = self.training_set_labels if labels is None else labels
        self.forward(data, False)
        self.backward(labels, 0, False)
        epsilon_mask = {}
        for layer in self.layers:
            if layer is self.layers.first:
                grad = layer.dW.reshape(-1, 1)
            else:
                grad = np.concatenate((grad, layer.dW.reshape(-1, 1)), axis=0)
            grad = np.concatenate((grad, layer.db.reshape(-1, 1)), axis=0)
            epsilon_mask[layer] = (np.zeros(layer.W.shape), np.zeros(layer.b.shape))
        grad_approximate = np.zeros((grad.shape[0], 1))
        index = 0
        for layer in self.layers:
            for i in range(layer.W.shape[0]):
                for j in range(layer.W.shape[1]):
                    epsilon_mask[layer][0][i, j] = epsilon
                    self.forward(data, False, epsilon_mask)
                    J_plus = self.compute_cost(labels)
                    epsilon_mask[layer][0][i, j] = -epsilon
                    self.forward(data, False, epsilon_mask)
                    J_minus = self.compute_cost(labels)
                    grad_approximate[index] = (J_plus - J_minus)/(2*epsilon)
                    index += 1
                    epsilon_mask[layer][0][i, j] = 0
            for i in range(layer.b.shape[0]):
                epsilon_mask[layer][1][i, 0] = epsilon
                self.forward(data, False, epsilon_mask)
                J_plus = self.compute_cost(labels)
                epsilon_mask[layer][1][i, 0] = -epsilon
                self.forward(data, False, epsilon_mask)
                J_minus = self.compute_cost(labels)
                grad_approximate[index] = (J_plus - J_minus)/(2*epsilon)
                index += 1
                epsilon_mask[layer][1][i, 0] = 0
        if np.linalg.norm(grad - grad_approximate)/(np.linalg.norm(grad) + np.linalg.norm(grad_approximate)) > epsilon:
            return "Correct"
        return "Incorrect"

    def generate_mini_batches(self, data: NDArray, labels: NDArray, mini_batch_size: int) -> list[tuple[NDArray, NDArray]]:
        size = data.shape[1]
        mini_batches = []
        permutation = list(np.random.permutation(size))
        shuffled_data = data[:, permutation]
        shuffled_labels = labels[:, permutation]
        for i in range(0, size, mini_batch_size):
            mini_batch_X = shuffled_data[:, i:i+mini_batch_size]
            mini_batch_Y = shuffled_labels[:, i:i+mini_batch_size]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        return mini_batches

    def update_learning_rate(self, learning_rate0: float, epoch: int, decay_rate: float, time_interval: int) -> float:
        return learning_rate0/(1+decay_rate*np.floor(epoch/time_interval))

    def train(
        self,
        training_set: NDArray = None,
        training_set_labels: NDArray = None,
        no_of_epochs: int = 1000,
        regularization: str = None,
        reg_lambd: float = 0,
        dropout: bool = False,
        optimization: dict = {'name': 'gradient descent'},
        is_mini_batch: bool = False,
        mini_batch_size: int = 64,
        is_learning_rate_decaying: bool = False,
        learning_rate: float = 0.01,
        learning_rate_decay_rate: float = 0.3,
        learning_rate_update_time_interval: int = 1000
    ) -> None:
        data = self.training_set if training_set is None else training_set
        labels = self.training_set_labels if training_set_labels is None else training_set_labels
        learning_rate0 = learning_rate
        for i in range(no_of_epochs):
            if is_learning_rate_decaying:
                learning_rate = self.update_learning_rate(
                    learning_rate0, i, learning_rate_decay_rate, learning_rate_update_time_interval
                    )
            if is_mini_batch:
                mini_batches = self.generate_mini_batches(data, labels, mini_batch_size)
                cost = 0
                for mini_batch in mini_batches:
                    self.forward(mini_batch[0], dropout)
                    self.backward(mini_batch[0], mini_batch[1], reg_lambd, dropout)
                    self.update_layers(optimization, learning_rate)
                    if optimization['name'] == 'adam':
                        optimization['t'] += 1
                    cost += self.compute_cost(mini_batch[1], regularization, reg_lambd)
                cost = cost/labels.shape[1]
            else:
                self.forward(data, dropout)
                self.backward(data, labels, reg_lambd, dropout)
                self.update_layers(optimization, learning_rate)
                cost = self.compute_cost(labels, regularization, reg_lambd)
            self.epoch += 1
            if self.epoch % 100 == 0 or i == no_of_epochs-1:
                self.costs[self.epoch] = cost

    def predict(self, data: NDArray):
        self.forward(data, False)
        return np.squeeze(self.layers.last.A)
