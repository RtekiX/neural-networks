# -*- coding: utf-8 -*-
'''
this python file validates a simple neural networks

dataset: audit risk data

attributes: [int, float], number: 17

label: [bool], valueset: [1, 0]

'''
from numpy.core.fromnumeric import mean
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("error", category=RuntimeWarning)


def sigmoid(z):
    """sigmoid activation function

    Parameters
    ----------
    z : [Number or Number List]

    Returns
    -------
    [Number or Number List]
    """    ''''''
    z[z < -100] = -100  # avoid overflow
    result = 1.0/(1 + np.exp(-1.0*z))
    return result


def MSEloss(y_predict, y_true):
    """calculate MSE loss

    Parameters
    ----------
    y_predict : [Number or Number List]
        [the label that model predicts]
    y_true : [Number or Number List]
        [true label]

    Returns
    -------
    [float]
        [return the average of all MSE loss]
    """    ''''''
    return np.mean((y_predict - y_true)**2)


def d_sigmoid(x):
    """derivative of sigmoid function

    Parameters
    ----------
    x : [Number or a Number list]
        [any]

    Returns
    -------
    [same as x]
        [any]
    """    ''''''
    return x*(1 - x)


def predict(w_input, b_input, w_hidden, b_hidden, test_feature, test_label):
    """given a parameter list, use test_data to calculate prediction accuracy

    Parameters
    ----------
    w_input : [M x N]
        [input layer <-> hidden layer]
    b_input : [1 x N]
        [input layer <-> hidden layer]
    w_hidden : [N x 1]
        [hidden layer <-> output layer]
    b_hidden : [float]
        [hidden layer <-> output layer]
    test_feature : [? x M]
        [the number of test_data is ?]
    test_label : [? x 1]
        [the number of test_data is ?]

    Returns
    -------
    [float]
        [the accuracy of prediction]
    """    ''''''
    test_feature = np.asarray(test_feature)
    test_label = np.asarray(test_label)

    y_prediction = []

    for i in range(0, len(test_feature)):
        input_x = np.asarray(test_feature[i]).reshape(
            1, len(train_feature[i]))
        y_true = test_label[i]
        # [1 x M]*[M x N] -> [1 x N]
        hidden_x = sigmoid(np.dot(input_x, w_input) + b_input)
        # [1 x N]*[N x 1] -> [1 x 1] -> float
        y_pred = np.sum(sigmoid(np.dot(hidden_x, w_hidden) + b_hidden))
        y_prediction.append(y_pred)

    y_prediction = np.asarray(y_prediction)
    y_true = test_label.astype(int).reshape(1, len(test_label))
    cost = MSEloss(y_prediction, y_true)
    y_prediction[y_prediction >= 0.5] = 1
    y_prediction[y_prediction < 0.5] = 0
    y_prediction = y_prediction.astype(int).reshape(1, len(y_prediction))

    predict_result = y_true - y_prediction
    right_num = np.sum(predict_result == 0)
    wrong_num = np.sum(predict_result != 0)
    acc = float(right_num)/(right_num + wrong_num)
    return acc, cost


if __name__ == "__main__":
    audit_dataset = pd.read_csv("./trial.csv")  # all data list
    audit_data_array = np.asarray(audit_dataset)  # transfrom to numpy list
    np.random.shuffle(audit_data_array)

    train_data = audit_data_array[0:700, :]
    train_feature = train_data[:, :-1]
    train_label = train_data[:, -1:]

    test_data = audit_data_array[-74:, :]
    test_feature = test_data[:, :-1]
    test_label = test_data[:, -1:]

    # use 1 hidden layer, 3~8 hidden nodes, defaut 3
    # input layer: 17 nodes, as M
    # hidden layer: 3-8 nodes, as N
    # output layer: 1 node
    input_node = 17
    hidden_node = 8
    print("hidden node: {}".format(hidden_node))

    # w[i, j] refers to the edge between input i node to hidden j node
    # input layer <-> hidden layer, w and b
    w_input = np.random.randn(input_node, hidden_node) / \
        np.sqrt(input_node)  # M x N
    b_input = np.random.random((1, hidden_node)) - 0.5  # 1 x N

    # hidden layer <-> output layer, w and b
    w_hidden = np.random.randn(hidden_node, 1) / np.sqrt(hidden_node)   # N x 1
    b_hidden = np.random.rand() - 0.5

    max_step = 1000
    learning_rate = 0.001
    print("max_step: {}, learning rate: {}".format(max_step, learning_rate))
    for step in range(0, max_step):
        for i in range(0, len(train_data)):
            input_x = np.asarray(train_feature[i]).reshape(
                1, len(train_feature[i]))
            y_true = train_label[i]
            # [1 x M]*[M x N] -> [1 x N]
            hidden_x = sigmoid(np.dot(input_x, w_input) + b_input)
            # [1 x N]*[N x 1] -> [1 x 1] -> float
            y_pred = np.sum(sigmoid(np.dot(hidden_x, w_hidden) + b_hidden))

            # update w and b
            dLdy_pred = np.sum(2*(y_pred - y_true))

            # [N x 1]*[1 x M] -> [N x M]
            dw_input = np.dot(
                np.dot(dLdy_pred*d_sigmoid(hidden_x)*d_sigmoid(y_pred), w_hidden), input_x)
            # dw_hidden -> [1 x N]
            dw_hidden = np.dot(dLdy_pred*d_sigmoid(y_pred), hidden_x)
            # db_input -> [N x 1]
            db_input = np.dot(dLdy_pred*d_sigmoid(hidden_x)
                              * d_sigmoid(y_pred), w_hidden)
            db_hidden = dLdy_pred*d_sigmoid(y_pred)

            w_input -= learning_rate*dw_input.T
            w_hidden -= learning_rate*dw_hidden.T
            b_input -= learning_rate*db_input.T
            b_hidden -= learning_rate*db_hidden

        if step % 100 == 0 or step == (max_step - 1):
            acc, cost = predict(w_input, b_input, w_hidden,
                                b_hidden, test_feature, test_label)
            print("Step:{}  Acc: {}  Cost: {}".format(step, acc, cost))
