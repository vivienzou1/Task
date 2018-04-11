#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 13:37:49 2018
@author: yinchenli
"""
import sys
import numpy as np
from random import shuffle
import copy
from datetime import datetime

switch = [{"Loan": 0, "Bank_Account": 1, "CD": 2, "Mortgage": 3, "Fund": 4},
          {"Business": 0, "Professional": 1, "Student": 2, "Doctor": 3, "Other": 4},
          {"Small": 0, "Medium": 1, "Large": 2},
          { "Full": 0, "Web&Email": 1, "Web": 2, "None": 3}]


sim_matrix = [
              np.array([
                        [ 1.0, 0.0, 0.1, 0.3, 0.2 ],
                        [ 0.0, 1.0, 0.0, 0.0, 0.0 ],
                        [ 0.1, 0.0, 1.0, 0.2, 0.2 ],
                        [ 0.3, 0.0, 0.2, 1.0, 0.1 ],
                        [ 0.2, 0.0, 0.2, 0.1, 1.0 ]
                        ]),
              np.array([
                        [ 1.0, 0.2, 0.1, 0.2, 0.0 ],
                        [ 0.2, 1.0, 0.2, 0.1, 0.0 ],
                        [ 0.1, 0.2, 1.0, 0.1, 0.0 ],
                        [ 0.2, 0.1, 0.1, 1.0, 0.0 ],
                        [ 0.0, 0.0, 0.0, 0.0, 1.0 ]
                        ]),

              np.array([
                        [ 1.0, 0.1, 0.0 ],
                        [ 0.1, 1.0, 0.1 ],
                        [ 0.0, 0.1, 1.0 ]
                       ]),
              np.array([
                        [ 1.0, 0.8, 0.0, 0.0 ],
                        [ 0.8, 1.0, 0.1, 0.5 ],
                        [ 0.0, 0.1, 1.0, 0.4 ],
                        [ 0.0, 0.5, 0.4, 1.0 ]
                        ])
              ]

weight = [0.05316756850374026, 0.02207398574536793, 4.296957469021434, 19.55068755364229, 0.16916465069104425, 0.00027251867418483347, 0.17466953632626236, 1.7128500572955707]

# Pram: file_path: string
# Return: dataset: [[data]], is_real[boolean]
def load_data(file_path):
    dataset = []
    is_real = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            if len(line.strip()) == 0 or line.startswith("@relation") or line.startswith("@data"):
                continue
            elif line.startswith("@attribute"):
                attr = line.strip()
                if attr.endswith("real"):
                    is_real.append(True)
                else:
                    is_real.append(False)
            else:
                dataset.append(line.replace('\n', '').replace('\r', '').split(','))
    return dataset, is_real

# Param: dataset, is_real
def normalize(dataset, is_real, low, up):
    normalized = []

    #make a deep copy of the original dataset
    copy_dataset = copy.deepcopy(dataset)
    for i in range(len(copy_dataset)):
        # the last attr is the label
        for j in range(len(is_real) - 1):
            if is_real[j]:
                (copy_dataset[i])[j] = (float((copy_dataset[i])[j]) - low[j]) / (up[j] - low[j])
        normalized.append(copy_dataset[i])
    return normalized

def calculate_min_max(dataset, is_real):

    copy_dataset = copy.deepcopy(dataset)
    length = len(is_real)
    low = [None] * length
    up = [None] * length

    for i in range(len(copy_dataset)):
         for j in range(length - 1):
              if i == 0:
                  if is_real[j]:
                      low[j] = float((copy_dataset[i])[j])
                      up[j] = float((copy_dataset[i])[j])

              else:
                  if is_real[j]:
                      low[j] = min(low[j], float((copy_dataset[i])[j]))
                      up[j] = max(up[j], float((copy_dataset[i])[j]))
    return low, up


# params: two data ponits, whether or not is real for each attr, and the weight for each attr
# return the (distance**2) between two data points
def caluclate_distance(train_point, test_point, is_real):
    # the last attr is the label or prediction
    length = len(is_real) - 1
    sum = 0
    j = 0
    for i in range(length):
        if is_real[i]:
            sum += weight[i] * (train_point[i] - test_point[i])**2
        else:
            sum += weight[i] * (1 - ((sim_matrix[j])[switch[j][train_point[i]]][switch[j][test_point[i]]]))
            j += 1
    return sum

# params: a train set and a test set
# return a list of the test size, each element is the k nb for the correspoding test data
def get_k_nb(trainset, testset, k, is_real):
    class Neighbor:
        def __init__(self, dist, data):
            self.dist = dist
            self.data = data

    trainset_copy = copy.deepcopy(trainset)
    testset_copy = copy.deepcopy(testset)

    k_neighbors_all = []
    for test in testset_copy:
        neighbors = []
        for train in trainset_copy:
            dis_sq = caluclate_distance(train, test, is_real)
            neighbor = Neighbor(dis_sq, train)
            neighbors.append(neighbor)
        k_neighbors = sorted(neighbors, key= lambda neighbor: neighbor.dist)[0:k]
        test_neighbor = []
        for e in k_neighbors:
            test_neighbor.append(e.data)
        k_neighbors_all.append(test_neighbor)
    return k_neighbors_all

def get_predictions(k_neighbors_all, k):
    k_neighbors_all_copy = copy.deepcopy(k_neighbors_all)
    predictions = []
    for test in k_neighbors_all_copy:
        prediction = 0
        for i in range(k):
            prediction += float((test[i])[-1])
        predictions.append(prediction/k)
    return predictions

        
def cross_validation(k, n, whole_set, is_real):
    whole_set_copy = copy.deepcopy(whole_set)
    shuffle(whole_set_copy)

    test_size = int(len(whole_set_copy) / n)
    performance = []
    for i in range(1, n + 1):
        # prepare the test set
        test_set = whole_set_copy[test_size * (i - 1):test_size * i]
        # prepare the train set
        train_set = []
        for e in whole_set_copy:
            if e not in test_set:
                train_set.append(e)
        k_neighbors = get_k_nb(train_set, test_set, k, is_real)
        predictions = get_predictions(k_neighbors, k)
        mse = 0
        for j in range(len(test_set)):
            mse += (predictions[j] - float((test_set[j])[-1]))**2
        performance.append(mse/(len(test_set)))
    return (sum(performance) / n), performance

# main
if __name__ == "__main__":
    file_path_train = sys.argv[1]
    file_path_test = sys.argv[2]
    k = int(sys.argv[3])
    n = int(sys.argv[4])

    result = "Summary of Result for productReal" + "\n\n"
    result += "run at " + str(datetime.now()) + "\n\n"

    result += "====== Prediction on the test set ======\n\n"
    train_set, is_real = load_data(file_path_train)
    test_set, not_important = load_data(file_path_test)
    whole_set = train_set + test_set
    low, up = calculate_min_max(whole_set, is_real)
    test_normalized = normalize(test_set, is_real, low, up)
    train_normalized = normalize(train_set,is_real, low, up)

    k_neighbours = get_k_nb(train_normalized, test_normalized, 5, is_real)
    predictions = get_predictions(k_neighbours, 5)
    for i in range(len(test_set)):
        result += "The predictions of " + str(test_set[i]) + " is " + str(predictions[i]) + '\n'
    result += '\n'

    result += "====== " + str(n) + " folds validation on train set ======\n\n"
    performance, average = cross_validation(k, n, train_normalized, is_real)
    result += "MSE for each time is " + str(performance) + '\n'
    result += "The average MSE is " + str(average) + '\n'

    # result += "\n\n"
    f = open('result.txt','a+')
    f.write(result)
    f.close()
    print ("Execution done, please see result in result.txt")