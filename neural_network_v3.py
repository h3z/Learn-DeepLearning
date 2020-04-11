'''
https://github.com/Kulbear/deep-learning-coursera/blob/master/Neural%20Networks%20and%20Deep%20Learning/Planar%20data%20classification%20with%20one%20hidden%20layer.ipynb
https://github.com/Kulbear/deep-learning-coursera/blob/master/Neural%20Networks%20and%20Deep%20Learning/Deep%20Neural%20Network%20-%20Application.ipynb
'''

import numpy as np
import matplotlib.pyplot as plt


def initializer_parameters(layer_dims):
    parameters = {}
    L = len(layer_dims)
    for i in range(1, L):
        parameters['W' + str(i)] = np.random.randn(layer_dims[i], layer_dims[i - 1]) * 0.01
        parameters['b' + str(i)] = np.zeros((layer_dims[i], 1))

    return parameters


def sigmoid(Z):
    activation_cache = Z
    A = 1. / (1 + np.exp(-Z))
    return A, activation_cache


def relu(Z):
    activation_cache = Z
    A = np.maximum(0, Z)
    return A, activation_cache


def linear_forward(A_prev, W, b):
    assert W.shape[1] == A_prev.shape[0]
    assert W.shape[0] == b.shape[0]
    linear_cache = (A_prev, W, b)
    return np.dot(W, A_prev) + b, linear_cache


def linear_activation_forward(A_prev, W, b, activation):
    Z, linear_cache = linear_forward(A_prev, W, b)
    if activation == 'sigmoid':
        A, activation_cache = sigmoid(Z)
    elif activation == 'relu':
        A, activation_cache = relu(Z)
    return A, (linear_cache, activation_cache)


def forward_propagate(X, parameters):
    L = len(parameters) // 2
    caches = []
    A = X
    for i in range(L - 1):
        l = str(i + 1)
        A, cache = linear_activation_forward(A, parameters['W' + l], parameters['b' + l], activation='relu')
        caches.append(cache)
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation='sigmoid')
    caches.append(cache)
    return AL, caches


def sigmoid_backward(dA, cache):
    Z = cache
    s, _ = sigmoid(Z)
    dZ = dA * s * (1 - s)
    assert dZ.shape == Z.shape
    return dZ


def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


def linear_backward(dZ, linear_cache):
    A_prev, W, b = linear_cache
    m = A_prev.shape[1]
    dW = 1. / m * np.dot(dZ, A_prev.T)
    db = 1. / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    return dW, db, dA_prev


def linear_activation_backward(dAL, cache, activation):
    linear_cache, activation_cache = cache

    if activation == 'sigmoid':
        dZ = sigmoid_backward(dAL, activation_cache)
    elif activation == 'relu':
        dZ = relu_backward(dAL, activation_cache)

    dW, db, dA_prev = linear_backward(dZ, linear_cache)
    return dW, db, dA_prev


def backward_propagate(Y, AL, caches):
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    L = len(caches)
    dW, db, dA_prev = linear_activation_backward(dAL, caches[L - 1], 'sigmoid')
    grads = {'dW' + str(L): dW,
             'db' + str(L): db}
    for i in reversed(range(1, L)):
        dW, db, dA_prev = linear_activation_backward(dA_prev, caches[i - 1], 'relu')
        grads['dW' + str(i)] = dW
        grads['db' + str(i)] = db
    return grads


def compute_cost(AL, Y):
    return -np.average(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for i in range(L):
        l = str(i + 1)
        parameters['W' + l] = parameters['W' + l] - learning_rate * grads['dW' + l]
        parameters['b' + l] = parameters['b' + l] - learning_rate * grads['db' + l]
    return parameters


def model(X, Y, layer_dims, learning_rate=0.0075, iteration_num=2500):
    assert layer_dims[0] == X.shape[0]

    parameters = initializer_parameters(layer_dims)
    costs = []

    for i in range(iteration_num):

        AL, caches = forward_propagate(X, parameters)

        cost = compute_cost(AL, Y)

        grads = backward_propagate(Y, AL, caches)

        parameters = update_parameters(parameters, grads, learning_rate)

        if i % 100 == 0:
            print(i, "cost: ", cost)
            costs.append(cost)

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.show()
    return parameters


from data.catvnoncat_util import load_data

train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_data()

X = train_set_x_orig.reshape((train_set_x_orig.shape[0], -1)).T
X = X / 255.
Y = train_set_y_orig

parameters = model(X, Y, [X.shape[0], 7, 1])
