import numpy as np
import h5py
from numpy.typing import NDArray
from function_utils import sigmoid
import logging

logging.basicConfig(
    filename='file.log', filemode='w', level=logging.DEBUG,
    format='%(asctime)s - %(message)s', datefmt='%H:%M:%S'
)


class LogisticRegression:
    def __init__(
        self,
        training_set: NDArray,
        training_set_labels: NDArray,
        test_set: NDArray,
        test_set_labels: NDArray,
        learning_rate: float = 0.01,
        threshold: float = 0.5
    ) -> None:
        self.X = training_set
        self.Y = training_set_labels
        self.X_test: NDArray = test_set
        self.Y_test = test_set_labels
        self.learning_rate = learning_rate
        self.threshold = threshold

        self.iteration = 0
        self.training_set_size = self.X.shape[1]
        self.w = np.zeros((self.X.shape[0], 1))
        self.b = 0.0
        self.costs = []

    def propagate(self) -> tuple[dict[str, NDArray], float]:
        A = sigmoid(np.dot(self.w.T, self.X) + self.b)
        cost = -np.sum(np.dot(self.Y, np.log(A).T)+np.dot(1-self.Y, np.log(1-A).T))/self.training_set_size
        dw = np.dot(self.X, (A-self.Y).T)/self.training_set_size
        db = np.sum(A-self.Y)/self.training_set_size
        cost = np.squeeze(np.array(cost))
        grads = {"dw": dw,
                 "db": db}
        return grads, cost

    def optimize(self, no_of_iterations: int) -> tuple[dict[str, NDArray], list[float]]:
        costs = []
        for _ in range(no_of_iterations):
            grads, cost = self.propagate()
            dw = grads["dw"]
            db = grads["db"]
            self.w = self.w - self.learning_rate*dw
            self.b = self.b - self.learning_rate*db
            self.iteration += 1
            if self.iteration % 100 == 0:
                logging.debug(f"iteration={self.iteration} - cost={cost}")
                costs.append(cost)
        return costs

    def predict(self, data_set: NDArray) -> NDArray:
        prediction = np.zeros((1, data_set.shape[1]))
        w = self.w.reshape(data_set.shape[0], 1)
        A = sigmoid(np.dot(w.T, data_set)+self.b)
        for i in range(A.shape[1]):
            prediction[0, i] = 1 if A[0, i] > self.threshold else 0
        return prediction

    def run(self, no_of_iterations: int) -> dict[str, list | NDArray | float]:
        self.costs.extend(self.optimize(no_of_iterations))
        prediction_training = self.predict(self.X)
        prediction_test = self.predict(self.X_test)
        return {"no_of_iterations": self.iteration,
                "costs": self.costs,
                "prediction_training": prediction_training,
                "prediction_test": prediction_test,
                "w": self.w,
                "b": self.b,
                "learning_rate": self.learning_rate,
                "threshold": self.threshold}


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
    lr = LogisticRegression(training_set, training_set_labels, test_set, test_set_labels)
    result = lr.run(1000)
    logging.debug(result)
