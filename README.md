# Task-11

This repo is working on the KNN classification and regression, as well as decision tree for classification. All the input data for training and testing are formatted in .arff.

All model are produced by `numpy` and read by `arff`, no other package related. 


## KNN
### Classification
Accuracy: 91% (weighted) to 90.4% (without weighted)

Python 3.6

implemented in Ipython jupyter notebook:
* Run `Knn_Classifier_91%.py` for KNN Classification

Cross validation is in 80% for training and 20% testing, from `trainProfSelection.arff`; although test code and label prediction is via `testProfSelection.arff`

### Binary
Accuracy: 91.25% 

Average weights, unweighted feature

Python 3.6
* Run `prodBinary.py` for KNN binary label by command line:
```bash
$ python productBinary.py 5 10 trainProdIntro.binary.arff testProdIntro.binary.arff
```

### Real
MSE = 14.89 with convergecy

weighted feature 

Python 3.6
* Run `prodReal.py` for KNN real prediction by command line:
```bash
$ python prodReal.py trainProdIntro.real.arff testProdIntro.real.arff 5 10
```


## Decision Tree
Accuracy of part A dataset: 85.5% 

Accuracy of part B datasetB: 95.6%
Python 2.7
* Run `decision_tree.py` for Desicion tree Classification, you can via command line:
```bash
$ python decision_tree.py trainProdSelection.arff testProdSelection.arff 15
```
Otherwise, use IDE and edit run configuration. 
