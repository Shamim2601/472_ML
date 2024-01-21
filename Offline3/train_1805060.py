
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
from torch.utils.data import TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import seaborn as sns
import pickle

class FNet:
    def __init__(self, layers_config):
        self.layers_config = layers_config
        self.weights = []
        self.bias = []
        self.average_weights = []
        self.average_bias = []
        self.eta = 0.005

    def initialize_weights(self):
        if len(self.layers_config) < 3:
            print("Incorrect network structure. Check the neural network layer configuration")
        else:
            layer_count = len(self.layers_config)
            self.weights.append([])
            self.bias.append([])
            self.average_weights.append([])
            self.average_bias.append([])
            for i in range(1, layer_count):
                neurons_previous = self.layers_config[i - 1]
                neurons_current = self.layers_config[i]
                single_layer_weights = np.random.normal(0, np.sqrt(2 / neurons_previous),
                                                        (neurons_current, neurons_previous))
                single_layer_bias = np.random.normal(0, np.sqrt(2 / neurons_previous), (neurons_current, 1))
                self.weights.append(single_layer_weights)
                self.bias.append(single_layer_bias)
                self.average_weights.append(single_layer_weights)
                self.average_bias.append(single_layer_bias)

    def one_hot_encode(self, y, num_classes=26):
        return np.eye(num_classes)[y]

    def ReLU(self, x):
        return np.maximum(0, x)

    def ReLU_derivative(self, values):
        result = [1 if x > 0 else 0 for x in values]
        return result


    def Softmax(self, x):
        return np.exp(x - np.max(x)) / (np.sum(np.exp(x - np.max(x))))

    def feedforward(self, input_image):
        a = []
        a.append(input_image.reshape(len(input_image), 1))
        for i in range(1, len(self.layers_config) - 1):
            a.append(self.ReLU((self.weights[i] @ a[i - 1]).reshape(len(self.weights[i]), 1) + self.bias[i]))
        y_hat = self.Softmax((self.weights[-1] @ a[-1]) + self.bias[-1]).reshape(len(self.weights[-1]),)
        a.append(y_hat)
        return a

    def backprop(self, a, ground_output_y):
        delta_error = list(np.empty_like(a))
        index_count = len(self.layers_config) - 1
        delta_error[index_count] = (a[index_count] - ground_output_y).reshape(len(a[index_count]), 1)
        self.average_bias[index_count] = self.average_bias[index_count] + delta_error[index_count]
        self.average_weights[index_count] = self.average_weights[index_count] + (
                    delta_error[index_count] @ a[index_count - 1].T)
        for i in range(index_count - 1, 0, -1):
            h_derivative = np.array(self.ReLU_derivative(a[i])).reshape(1, len(a[i])) * np.eye(len(a[i]))
            delta_error[i] = h_derivative.T @ self.weights[i + 1].T @ delta_error[i + 1]
            self.average_bias[i] = self.average_bias[i] + delta_error[i]
            self.average_weights[i] = self.average_weights[i] + (delta_error[i] @ a[i - 1].T)

    def train(self, X_train, y_train, epoch_size=10, batch_size=4096):
        batch_numbers = int(len(X_train) / batch_size)

        # Single Sample Updates
        shuffle_order = np.random.permutation(len(X_train))
        for i in range(1, epoch_size + 1):
            print("Running Epoch {}".format(i))
            shuffle_order = np.random.permutation(len(X_train))
            for j in range(batch_numbers):
                print("Running batch number {} in epoch {}".format(j+1, i))
                for k in range(batch_size):
                    shuffle_index = j * batch_size + k
                    sample_x = X_train[shuffle_index, :]
                    sample_y = y_train[shuffle_index, :]
                    a_values = self.feedforward(sample_x)
                    self.backprop(a_values, sample_y)
                for a in range(len(self.weights)):
                    value_weight = self.weights[a]
                    value_average_weight = np.multiply(self.average_weights[a], (self.eta / batch_size))
                    self.weights[a] = np.subtract(value_weight, value_average_weight)
                    value_bias = self.bias[a]
                    value_average_bias = np.multiply(self.average_bias[a], (self.eta / batch_size))
                    self.bias[a] = np.subtract(value_bias, value_average_bias)



def read_and_process():
    train_validation_dataset = datasets.EMNIST(root='./data', split='letters',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)

    independent_test_dataset = datasets.EMNIST(root='./data',
                                               split='letters',
                                               train=False,
                                               transform=transforms.ToTensor())

    # Convert it to a list to use train_test_split
    train_dataset_list = list(train_validation_dataset)
    test_dataset_list = list(independent_test_dataset)

    # Extract features (X) and labels (y)
    X = np.array([item[0].numpy().flatten() for item in train_dataset_list])
    y = np.array([item[1] for item in train_dataset_list])
    X_test = np.array([item[0].numpy().flatten() for item in test_dataset_list])
    y_test = np.array([item[1] for item in test_dataset_list])

    # Split the dataset
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

    return X_train, X_val, y_train, y_val, X_test, y_test


def predict(network, X):
  y_output = np.array([network.feedforward(X[m, :])[len(network.layers_config) - 1] for m in range(len(X))])
  class_output = np.argmax(y_output, axis=1)

  return class_output

def evaluate_classification(y_true, y_pred):
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred) * 100
    print(f'Accuracy: {accuracy:.4f}')

    # F1 Score
    f1 = f1_score(y_true, y_pred, average='weighted')
    print(f'F1 Score: {f1:.4f}')

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print('Confusion Matrix:')
    # print(cm)

    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()



def save_weights_bias(network, file_path='weights_bias.pkl'):
    weights_bias_dict = {'weights': network.weights, 'bias': network.bias}
    with open(file_path, 'wb') as file:
        pickle.dump(weights_bias_dict, file)
    print(f'Weights and biases saved to {file_path}')




# --------- main function to load data and train --------------
if __name__ == "__main__":
    X_train, X_val, y_train, y_val, X_test, y_test = read_and_process()

    y_train -= 1
    y_val -= 1
    y_test -= 1

    network = FNet(layers_config=[784, 256, 26])
    network.initialize_weights()
    network.train(X_train, network.one_hot_encode(y_train), epoch_size=10, batch_size=13260)

    y_pred = predict(network, X_val)

    evaluate_classification(y_val, y_pred)

    save_weights_bias(network, "model2_1805060.pkl")