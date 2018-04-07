# Task-11

This repo is working on the KNN classification and regression, as well as decision tree for classification, expressed by python 3.6 for KNN and python 2.7 for Decision Tree. All the input data for training and testing are formatted in .arff.

All model are produced by `numpy` and read by `arff`, no other package related. 


## KNN

To train the model, we suggested in ipython by jupyter notebook:
* Run `Knn_Classifier_91%.py` for KNN Classification

Cross validation is in 80% for training and 20% testing, from `trainProfSelection.arff`; although test code and label prediction is via `testProfSelection.arff`

The global accuracy for the KNN classifier is around 91% (weighted) to 90.4% (without weighted) depended on convergence degree, we suggest at least in hundreds of iteration. 


## Decision Tree

* Run `decision_tree.py` for Desicion tree Classification, you can:
1. Use command line:
* Run `python decision_tree.py <path_train_set> <path_test_set> <k_folds_for_cross_validation>`
2. Use IDE and edit run configuration
