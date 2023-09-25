import numpy as np
from numpy.typing import NDArray, ArrayLike


def sigmoid(x: NDArray) -> ArrayLike:
    return 1/(1+np.exp(-x))
