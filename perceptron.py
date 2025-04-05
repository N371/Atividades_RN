import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = np.zeros(input_size + 1)  # Inicializa os pesos com zeros
        self.learning_rate = learning_rate

    def predict(self, input_vector):
        activation = np.dot(input_vector, self.weights[1:]) + self.weights[0] #corrigido self.weights
        return 1 if activation >= 0 else -1

    def train(self, training_inputs, labels, num_epochs):
        errors = []
        for epoch in range(num_epochs): #corrigido num_epoch
            error = 0
            for input_vector, label in zip(training_inputs, labels):
                prediction = self.predict(input_vector)
                error += int(label != prediction)
                self.weights[1:] += self.learning_rate * (label - prediction) * input_vector #adicionado input_vector
                self.weights[0] += self.learning_rate * (label - prediction) #adicionado bias update
            errors.append(error)
        self.plot_errors(errors, num_epochs)

    def plot_errors(self, errors, num_epochs):
        plt.plot(range(1, num_epochs + 1), errors)
        plt.xlabel('Epoch')
        plt.ylabel('Number of errors')
        plt.title('Training errors over epochs')
        plt.show()

    def plot_data(self, training_inputs, labels):
        plt.figure(figsize=(8, 8))
        plt.scatter(training_inputs[:, 0], training_inputs[:, 1], c=labels, cmap='bwr')
        x_min, x_max = training_inputs[:, 0].min() - 1, training_inputs[:, 0].max() + 1
        y_min, y_max = training_inputs[:, 1].min() - 1, training_inputs[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
        z = np.array([self.predict(np.array([x, y])) for x, y in np.c_[xx.ravel(), yy.ravel()]]) #corrigido yy.rate() para yy.ravel()
        z = z.reshape(xx.shape)
        plt.contour(xx, yy, z, levels=[0], colors='k') #corrigido Z para z
        plt.title('Classification of Data Points')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()

np.random.seed(0)
training_inputs = np.random.randn(100, 2)
labels = np.array([1 if np.dot(x, [1, 2]) + 0.5 > 0 else -1 for x in training_inputs])

perceptron = Perceptron(input_size=2)
perceptron.train(training_inputs, labels, num_epochs=20)
perceptron.plot_data(training_inputs, labels)
