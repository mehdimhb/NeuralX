# NeuralX

NeuralX is a Python package for creating neural networks with a wide range of features, including gradient checks, regularization, dropout, mini batch, and various optimization methods. This package is designed to be beginner-friendly, making it easy for users to understand neural network structures and build powerful models.

## Features

- **Gradient Checks:** Ensure the correctness of your neural network's gradients with built-in gradient check tools.
- **Regularization:** Apply L1 and L2 regularization to prevent overfitting and improve model robustness.
- **Dropout:** Implement dropout layers for more effective regularization and enhanced generalization.
- **Optimization Methods:** Choose from different optimization methods, including the popular Adam optimizer, to train your models efficiently.

## Installation

You can install NeuralX using `pip`:

```bash
pip install neuralx
```

## Usage

To get started with NeuralX, follow these steps:

1. Import the necessary modules from NeuralX.
2. Build your neural network architecture, specifying layers, activation functions, and other parameters.
3. Train your model on your dataset using the chosen optimization method.
4. Evaluate the model's performance using the provided tools, including the confusion matrix implementation.

Here's a simple example of how to use NeuralX to create a basic neural network:

```python
import neuralx as nx

# create layers description
layers = [
        (training_set.shape[0], None, None, None),
        (5, 'he', 'relu', 1),  # no_of_units, normalization_method, activation_function, dropout_keeping_probability
        (2, 'he', 'relu', 1),
        (1, 'he', 'sigmoid', 1)
    ]

# Create a neural network
model = nx.NeuralNetwork(layers)

# Train the model
model.train(
    training_set,
    training_set_labels,
    no_of_epochs=1000,
    optimization={'name': 'momentum', 'beta': 0.9},
    is_mini_batch=True,
    learning_rate=0.001
)

# Evaluate the model using the confusion matrix
cm = nx.ConfusionMatrix(model, test_set, test_set_labels)
print(cm)
print(cm.statistics())
```

## Contributing

We welcome contributions from the community. Feel free to submit bug reports, feature requests, or even pull requests. Together, we can make NeuralX even better!

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.