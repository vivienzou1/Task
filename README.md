# Task-11

This repo is working on the KNN classification and regression, as well as decision tree for classification, expressed by python 3.6. All the input data for training and testing are formatted in .arff.

All model are produced by `numpy` and read by `arff`, no other package related. 


======== KNN ============

To train the model, we suggested in ipython by jupyter notebook:
* Run `Knn_Classifier_91%.py` for KNN Classification
* Run `decision_tree.py` for Desicion tree Classification

The global accuracy for the KNN classifier is around 91% to 90.4% depended on convergence degree, we suggest at least in hundreds of iteration. 


===== Decision Tree =====

To run the program, you can:
1. Use command line:
python decision_tree.py <path_train_set> <path_test_set> <k_folds_for_cross_validation>
2. Use IDE and edit run configuration
