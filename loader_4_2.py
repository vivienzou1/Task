#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 13:37:49 2018
@author: yinchenli
"""

class Customer:
    'common class for all types of customer'

    def __init__(self, inputString, Train):
        array = inputString.split(',')
        self.Ctype = array[0]
        self.lifestyle = array[1]
        self.vacation = float(array[2])
        self.eCredit = float(array[3])
        self.salary = float(array[4])
        self.prop = float(array[5].strip())
        self.cla = str(array[-1].strip())
        self.sim = float(0)
        if (Train == True):
            Knn.recordTrainMinMax(2, float(array[2]))
            Knn.recordTrainMinMax(3, float(array[3]))
            Knn.recordTrainMinMax(4, float(array[4]))
            Knn.recordTrainMinMax(5, float(array[5].strip()))
        else:
            Knn.recordTestMinMax(2, float(array[2]))
            Knn.recordTestMinMax(3, float(array[3]))
            Knn.recordTestMinMax(4, float(array[4]))
            Knn.recordTestMinMax(5, float(array[5].strip()))

    def __str__(self):
        return "this customer: type %s, life %s, vac %f eCredit %f salary %f prop %f class %s sim %f" % \
    (self.Ctype, self.lifestyle, self.vacation, self.eCredit, self.salary, self.prop, self.cla, self.sim)


class Knn:
    'main class to operate all'
    listTrainMin = [None]*6
    listTrainMax = [None]*6
    listTestMin = [None]*6
    listTestMax = [None]*6
    listTrainCustomer = []
    listTestCustomer = []

    def readFromFile_Train(name):
        with open(name, "r") as f:
            for line in f.readlines():
                if not line.startswith("@") and ',' in line:
                    Knn.listTrainCustomer.append(Customer(line, True))


    def readFromFile_Test(name):
        with open(name, "r") as f:
            for line in f.readlines():
                if not line.startswith("@") and ',' in line:
                    Knn.listTestCustomer.append(Customer(line, False))



    def initializeTrain():
        i = 3
        for i in range(len(Knn.listTrainMin)):
            Knn.listTrainMin[i]= float('inf')
        for i in range(len(Knn.listTrainMax)):
            Knn.listTrainMax[i] = float('-inf')

    def initializeTest():
        i = 3
        for i in range(len(Knn.listTestMin)):
            Knn.listTestMin[i]= float('inf')
        for i in range(len(Knn.listTestMax)):
            Knn.listTestMax[i] = float('-inf')

    def recordTrainMinMax(i, value):
        Knn.listTrainMin[i] = min(Knn.listTrainMin[i], value)
        Knn.listTrainMax[i] = max(Knn.listTrainMax[i], value)

    def recordTestMinMax(i, value):
        Knn.listTestMin[i] = min(Knn.listTestMin[i], value)
        Knn.listTestMax[i] = max(Knn.listTestMax[i], value)

    def normalizeTrain(feaIndex, OriValue):
        normValue = (OriValue - Knn.listTrainMin[feaIndex]) / (Knn.listTrainMax[feaIndex] - Knn.listTrainMin[feaIndex])
        return normValue

    def normalizeTest(feaIndex, OriValue):
        normValue = (OriValue - Knn.listTestMin[feaIndex]) / (Knn.listTestMax[feaIndex] - Knn.listTestMin[feaIndex])
        return normValue

Knn.initializeTrain()
Knn.readFromFile_Train('trainProdSelection.arff')

Knn.initializeTest()
Knn.readFromFile_Test('testProdSelection.arff')

# normalization
for i in range(len(Knn.listTrainCustomer)):
    t = Knn.listTrainCustomer[i]
#     print('before:')
#     print(t)
    t.eCredit = Knn.normalizeTrain(3, t.eCredit)
    t.vacation = Knn.normalizeTrain(2, t.vacation)
    t.salary = Knn.normalizeTrain(4, t.salary)
    t.prop = Knn.normalizeTrain(5, t.prop)
#     print('after:')
#     print(t)

for i in range(len(Knn.listTestCustomer)):
    y = Knn.listTestCustomer[i]
#     print('before:')
#     print(t)
    y.eCredit = Knn.normalizeTrain(3, y.eCredit)
    y.vacation = Knn.normalizeTrain(2, y.vacation)
    y.salary = Knn.normalizeTrain(4, y.salary)
    y.prop = Knn.normalizeTrain(5, y.prop)
#     print('after:')
#     print(t)

print(Knn.listTestCustomer[19])
