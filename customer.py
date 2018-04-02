#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 13:37:49 2018

@author: yinchenli
"""

class Customer:
    'common class for all types of customer'
    
    def __init__(self, inputString):
         array = inputString.split(',')        
         self.Ctype = array[0]
         self.lifestyle = array[1]
         self.vacation = float(array[2])
         Knn.recordMinMax(2, float(array[2]))
         self.eCredit = float(array[3])
         Knn.recordMinMax(3, float(array[3]))
         self.salary = float(array[4])
         Knn.recordMinMax(4, float(array[4]))
         self.prop = float(array[5].strip())
         Knn.recordMinMax(5, float(array[5].strip()))

    def __str__(self):
        return "this customer: type %s, life %s, vac %f eCredit %f salary %f prop %f " % (self.Ctype, self.lifestyle, self.vacation, self.eCredit, self.salary, self.prop)
   
        
class Knn:
    'main class to operate all'
    listMin = [None]*6
    listMax = [None]*6
    listTrainCustomer = []
    
    def readFromFile(name):
        with open(name, "r") as f:         
            for line in f.readlines():
                if not line.startswith("@") and ',' in line:
                    Knn.listTrainCustomer.append(Customer(line))
    
    def initialize():
        i = 3
        for i in range(len(Knn.listMin)):
            Knn.listMin[i]= float('inf')
        for i in range(len(Knn.listMax)):
            Knn.listMax[i] = float('-inf')
         
    def recordMinMax(i, value):
        Knn.listMin[i] = min(Knn.listMin[i], value)
        Knn.listMax[i] = max(Knn.listMax[i], value)
        
    def normalize(feaIndex, OriValue):
        normValue = (OriValue - Knn.listMin[feaIndex]) / (Knn.listMax[feaIndex] - Knn.listMin[feaIndex])
        return normValue
               
Knn.initialize()
Knn.readFromFile('trainProdSelection.arff')
for i in range(len(Knn.listTrainCustomer)):
    t = Knn.listTrainCustomer[i]
    print('before:')
    print(t)
    t.eCredit = Knn.normalize(3, t.eCredit)
    t.vacation = Knn.normalize(2, t.vacation)
    t.salary = Knn.normalize(4, t.salary)
    t.prop = Knn.normalize(5, t.prop)
    print('after:')
    print(t)
    

                  
    