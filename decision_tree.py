# !/usr/bin/python
from math import log
import operator
import random
import json
import sys

features_copy = []

def loadData(file_path):
    dataset = []
    features = []
    feature_continuous = []

    with open(file_path, 'rb') as f:
        for line in f.readlines():
            if len(line.strip()) == 0 or line.startswith("@relation") or line.startswith("@data"):
                continue
            elif line.startswith("@attribute"):
                vals = line.strip().split(' ')
                features.append(vals[1])
                is_real = True if vals[2] == "real" else False
                feature_continuous.append(is_real)
            else:
                line = line.strip().split(',')
                dataset.append(line)

    return dataset, features, feature_continuous


def generateTree(dataset, features, feature_continuous):
    labels = [example[-1] for example in dataset]

    if len(set(labels)) == 1:
        return labels[0]  # only have one lable, just return it

    if len(dataset[0]) == 1:
        return majorityCnt(labels)  # only 1 attribute in dataset, return the major label

    best_feature_index, best_split_point = get_best_split_feature(dataset, feature_continuous)
    if best_feature_index == -1:
        return majorityCnt(labels)  # can't find a split point, stop here

    split_feature = features[best_feature_index]

    root = {split_feature: {}}  # build tree in json format

    del (features[best_feature_index])

    if feature_continuous[best_feature_index]:
        # split continuous feature
        del (feature_continuous[best_feature_index])
        children_nodes = ['< ' + str(best_split_point), '>= ' + str(best_split_point)]
        lower_dataset, greater_dataset = split_continous(dataset, best_feature_index, best_split_point)
        root[split_feature][children_nodes[0]] = generateTree(lower_dataset, features[:], feature_continuous[:])
        root[split_feature][children_nodes[1]] = generateTree(greater_dataset, features[:], feature_continuous[:])

    else:
        # split nominal feature
        del (feature_continuous[best_feature_index])
        children_nodes = [example[best_feature_index] for example in dataset]
        for node in children_nodes:
            root[split_feature][node] = generateTree(split_nominal(dataset, best_feature_index, node), features[:],
                                                     feature_continuous[:])

    return root


def predict_label(sample, root):
    if not isinstance(root, dict):
        return root
    index = root.keys()[0]
    root = root[index]
    keys = root.keys()

    if '<' in keys[0] or '>=' in keys[0]:
        breakpoint = keys[0].split(' ')[-1]
        if sample[features_copy.index(index)] >= float(breakpoint):
            root = root['>= ' + breakpoint]
        else:
            root = root['< ' + breakpoint]
    else:
        root = root[sample[features_copy.index(index)]]
    return predict_label(sample, root)


def get_best_split_feature(dataset, feature_continuous):
    feature_cnt = len(dataset[0]) - 1
    curr_entropy = calculate_entropy(dataset)  # root entropy
    # print "Entropy: " + str(curr_entropy) + ", dataset: " + str(dataset)
    best_feature = -1
    best_split_point = 0.0
    best_info_gain = 0.0

    for i in range(feature_cnt):
        curr_feature_vals = [example[i] for example in dataset]

        if feature_continuous[i]:
            sorted_feature = sorted(set(curr_feature_vals))
            for value in sorted_feature:
                new_entropy = 0.0
                sub_dataset1, sub_dataset2 = split_continous(dataset, i, value)
                new_entropy += len(sub_dataset1) / float(len(dataset)) * calculate_entropy(sub_dataset1)
                new_entropy += len(sub_dataset2) / float(len(dataset)) * calculate_entropy(sub_dataset2)
                info_gain_ration = (curr_entropy - new_entropy) / curr_entropy
                if info_gain_ration >= best_info_gain:
                    best_info_gain = info_gain_ration
                    best_feature = i
                    best_split_point = value
        else:
            new_entropy = 0.0
            for value in set(curr_feature_vals):
                sub_dataset = split_nominal(dataset, i, value)
                probability = len(sub_dataset) / float(len(dataset))
                new_entropy += probability * calculate_entropy(sub_dataset)
            info_gain_ration = (curr_entropy - new_entropy) / curr_entropy
            if info_gain_ration >= best_info_gain:
                best_info_gain = info_gain_ration
                best_feature = i

    return best_feature, best_split_point


def split_nominal(dataset, index, value):
    sub_dataset = []
    for featVec in dataset:
        if featVec[index] == value:
            reducedFeatVec = featVec[:index]  # chop out axis used for splitting
            reducedFeatVec.extend(featVec[index + 1:])
            sub_dataset.append(reducedFeatVec)
    # print "Split nominal feature: " + features[index] + ": " + value
    # print "sub_dataset: " + str(sub_dataset) +'\n'
    return sub_dataset


def split_continous(dataset, index, value):
    sub_dataset1 = [vec[:index] + vec[index + 1:] for vec in dataset if float(vec[index]) < float(value)]
    sub_dataset2 = [vec[:index] + vec[index + 1:] for vec in dataset if float(vec[index]) >= float(value)]
    # print "Split continuous feature: " + features[index] + ": " + value
    # print "sub_dataset1: " + str(sub_dataset1)
    # print "sub_dataset2: " + str(sub_dataset2)  +'\n'
    return sub_dataset1, sub_dataset2


def calculate_entropy(data_set):
    size = len(data_set)
    label_cnt = {}
    for example in data_set:
        label = example[-1]
        if label in label_cnt:
            label_cnt[label] += 1
        else:
            label_cnt[label] = 1
    entropy = 0.0
    for key in label_cnt:
        probability = float(label_cnt[key]) / size
        entropy -= probability * log(probability, 2)

    return entropy


def majorityCnt(labels):
    classCount = {}
    for vote in labels:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


if __name__ == "__main__":
    # file_path = sys.argv[1]
    file_path = "trainProdSelection.arff"
    dataset, features, feature_continuous = loadData(file_path)
    features_copy = features[:]
    print "Loaded dataset: " + str(dataset) + '\n'
    print "features: " + str(features) + '\n'
    print "feature_continuous: " + str(feature_continuous) + '\n'

    tree = generateTree(dataset, features, feature_continuous)

    print json.dumps(tree, indent=4)

    sample_data_set, sample_feature, sample_feature_continues = loadData("testProdSelection.arff");
    for row in sample_data_set:
        print "The prediction of " + str(row) + " is " + predict_label(row, tree)