import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

    def plot_decision_boundary(self, X, y):
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
        Z = np.array([self.predict(np.array([x, y])) for x, y in np.c_[xx.ravel(), yy.ravel()]])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.bwr)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.bwr, s=20, edgecolors='k')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Decision Boundary')

    def plot_data_with_boundary(self, training_inputs, labels):
        plt.figure(figsize=(8, 8))
        self.plot_decision_boundary(training_inputs, labels)
        plt.title('Classification with Decision Boundary')
        plt.show()

if __name__ == "__main__":
    # Carregar dados de treino
    try:
        train_df = pd.read_csv('train_dataset1.csv')
        train_inputs = train_df[['x1', 'x2']].values.astype(float)  # Converter para array NumPy de floats
        train_labels = train_df['label'].values.astype(int)    # Converter para array NumPy de inteiros
        print("Dados de treino carregados com sucesso de train_dataset1.csv")
    except FileNotFoundError:
        print("Erro: Arquivo train_dataset1.csv não encontrado.")
        exit()

    # Carregar dados de teste
    try:
        test_df = pd.read_csv('test_dataset1.csv')
        test_inputs = test_df[['x1', 'x2']].values.astype(float)   # Converter para array NumPy de floats
        test_labels = test_df['label'].values.astype(int)     # Converter para array NumPy de inteiros
        print("Dados de teste carregados com sucesso de test_dataset1.csv")
    except FileNotFoundError:
        print("Erro: Arquivo test_dataset1.csv não encontrado.")
        exit()

    # Inicializar e treinar o Perceptron
    input_size = train_inputs.shape[1]
    perceptron = Perceptron(input_size=input_size, learning_rate=0.000000000000001) # Ajuste a taxa de aprendizado conforme necessário
    num_epochs = 100 # Ajuste o número de épocas conforme necessário
    perceptron.train(train_inputs, train_labels, num_epochs=num_epochs)
    perceptron.plot_data_with_boundary(train_inputs, train_labels)

    # Avaliar o Perceptron nos dados de teste
    correct_predictions = 0
    for input_vector, label in zip(test_inputs, test_labels):
        prediction = perceptron.predict(input_vector)
        if prediction == label:
            correct_predictions += 1

    accuracy = correct_predictions / len(test_labels)
    print(f"Acurácia no conjunto de teste: {accuracy * 100:.2f}%")

    # Opcional: Plotar os dados de teste com a fronteira de decisão aprendida
    plt.figure(figsize=(8, 8))
    perceptron.plot_decision_boundary(test_inputs, test_labels)
    plt.title('Classificação do Conjunto de Teste com Fronteira de Decisão')
    plt.show()
