import numpy as np
from numpy.typing import NDArray


def sigmoid(Z: NDArray) -> NDArray:
    return 1/(1+np.exp(-Z))


def relu(Z: NDArray) -> NDArray:
    return np.maximum(0, Z)


def relu_backward(dA: NDArray, Z: NDArray) -> NDArray:
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


def sigmoid_backward(dA: NDArray, Z: NDArray) -> NDArray:
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    return dZ
