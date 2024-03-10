import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import argparse

width = 100
height = 100

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cost(A, Y):
    return -np.sum(np.multiply(Y, np.log(A + 1e-6)) + np.multiply((1 - Y), np.log(1 - A + 1e-6)))


def mse(A, Y):
    return np.sum(np.power(A - Y, 2))


def predict(test_img, weights, bias):
    test_images, Y = load_dataset(test_img)

    A = sigmoid(np.dot(weights.T, test_images) + bias)
    print(f"Accuracy for test set: {100 - np.sum(abs(Y - A)) / len(Y)}%")


def load_dataset(data_folder):
    training_dir = os.fsencode(data_folder)
    training_images_dict = dict()

    for file in os.listdir(training_dir):
        filename = os.fsdecode(file)

        training_images_dict[filename] = np.array(
            Image.open(data_folder + filename).resize((width, height)))

    Y = []
    for i in training_images_dict.keys():
        if i.find("cat") != -1:
            Y.append(1)
        else:
            Y.append(0)

    training_images = list(training_images_dict.values())
    training_images = np.array(training_images)
    training_images = training_images.reshape(
        (training_images.shape[1] * training_images.shape[2] * 3, training_images.shape[0]))
    training_images = training_images / 255
    Y = np.array(Y).reshape(1, len(Y))

    return training_images, Y


def init_parameters(shape):

    weights = np.zeros((shape, 1)) + 1e-6
    bias = 0.0
    return weights, bias


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='Image classifier',
                                     description='Simple linear regression for image classification with L2 regularization')

    parser.add_argument('-c', '--cycles')
    parser.add_argument('-l', '--learning_rate')

    args = parser.parse_args()
    learning_rate = float(args.learning_rate)
    cycles = int(args.cycles)

    training_images, Y = load_dataset("C:\\datasets\\cats\\training-set\\")
    weights, bias = init_parameters(training_images.shape[0])
    dw, db = 0, 0
    m = training_images.shape[1]
    costs = []

    for _ in range(cycles):
        A = sigmoid(np.dot(weights.T, training_images) + bias)
        if _ % 100 == 0:
            print(f"prediction-{_}")

        loss_value = cost(A, Y) / m
        costs.append(loss_value)

        dw += np.dot(training_images, (A - Y).T) / m
        db += np.sum(A - Y) / m

        weights = weights - learning_rate * dw
        bias = bias - learning_rate * db

    predict("C:\\NN\\LR\\test1\\", weights, bias)

    plt.plot(costs, 'green')
    plt.title("Cost function")
    plt.show()

    print(f"MSE: {np.sum(costs) / m}")
    print(f"Accuracy for training set: {100 - np.sum(abs(A - Y)) / m}%")