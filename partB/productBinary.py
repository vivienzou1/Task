#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 13:37:49 2018
@author: yinchenli
"""
import sys
import numpy as np
from random import shuffle

serSwitcher = {"Loan": 0, "Bank_Account": 1, "CD": 2, "Mortgage": 3, "Fund": 4}
cusSwitcher = {"Business": 0, "Professional": 1, "Student": 2, "Doctor": 3, "Other": 4}
sizeSwitcher = {"Small": 0, "Medium": 1, "Large": 2}
promoSwitcher = { "Full": 0, "Web&Email": 1, "Web": 2, "None": 3}

class Product:
    'common class for all types of products'
    serTypeDistance = np.array([
        [ 1.0, 0.0, 0.1, 0.3, 0.2 ],
        [ 0.0, 1.0, 0.0, 0.0, 0.0 ],
        [ 0.1, 0.0, 1.0, 0.2, 0.2 ],
        [ 0.3, 0.0, 0.2, 1.0, 0.1 ],
        [ 0.2, 0.0, 0.2, 0.1, 1.0 ]
    ])
    
    cusTypeDistance = np.array([
        [ 1.0, 0.2, 0.1, 0.2, 0.0 ],
        [ 0.2, 1.0, 0.2, 0.1, 0.0 ],
        [ 0.1, 0.2, 1.0, 0.1, 0.0 ],
        [ 0.2, 0.1, 0.1, 1.0, 0.0 ],
        [ 0.0, 0.0, 0.0, 0.0, 1.0 ]
    ])
    
    sizeDistance = np.array([
        [ 1.0, 0.1, 0.0 ],
        [ 0.1, 1.0, 0.1 ],
        [ 0.0, 0.1, 1.0 ]
    ])
    
    promoDistance = np.array([
        [ 1.0, 0.8, 0.0, 0.0 ],
        [ 0.8, 1.0, 0.1, 0.5 ],
        [ 0.0, 0.1, 1.0, 0.4 ],
        [ 0.0, 0.5, 0.4, 1.0 ]
    ])

    def __init__(self, ser, cus, mFee, budget, size, promo, interest, period, label):

        self.ser = ser       
        self.cus = cus
        self.mFee = float(mFee)
        self.budget = float(budget)
        self.size = size
        self.promo = promo
        self.interest = float(interest)
        self.period = float(period)
        self.label = float(label)
        self.sim = float(0)
        self.predL = float(-1.0)
    
    def __str__(self):
        return "serType {} customer {} mFee {} budget {} size {} promo {} interest {} period {} sim {} label {} predicted {}".format \
    (self.ser, self.cus, self.mFee, self.budget, self.size, self.promo, self.interest, self.period, self.sim, self.label, self.predL)

        
class Knn:
    'main class to operate all'
    listTrainMin = [None]*8
    listTrainMax = [None]*8
    listTestMin = [None]*8
    listTestMax = [None]*8
    listTrainCustomer = []
    listTestCustomer = []
    
    def readFromFile_Train(self, name):
        with open(name, "r") as f:         
            for line in f.readlines():
                if not line.startswith("@") and ',' in line:
                    array = line.split(',')
                    ser = serSwitcher[array[0]]
                    cus = cusSwitcher[array[1]]
                    mFee = float(array[2])
                    budget = float(array[3])
                    size = sizeSwitcher[array[4]]
                    promo = promoSwitcher[array[5]]
                    interest = float(array[6])
                    period = float(array[7])
                    label = float(array[8].strip())
                    Knn.listTrainCustomer.append(Product(ser, cus, mFee, budget, size, promo, interest, period, label))
        return Knn.listTrainCustomer

    def readFromFile_Test(name):
        with open(name, "r") as f:
            for line in f.readlines():
                if not line.startswith("@") and ',' in line:
                    array = line.split(',')
                    ser = serSwitcher[array[0]]
                    cus = cusSwitcher[array[1]]
                    mFee = float(array[2])
                    budget = float(array[3])
                    size = sizeSwitcher[array[4]]
                    promo = promoSwitcher[array[5]]
                    interest = float(array[6])
                    period = float(array[7])
                    label = float(array[8].strip())
                    Knn.listTestCustomer.append(Product(ser, cus, mFee, budget, size, promo, interest, period, label))
                    
                    
    def initializeTrain():
        for i in range(len(Knn.listTrainMin)):
            Knn.listTrainMin[i]= float('inf')
        for i in range(len(Knn.listTrainMax)):
            Knn.listTrainMax[i] = float('-inf')
            
    def initializeTest():
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
        #print("000", OriValue)
        #print("1111", Knn.listTrainMin[feaIndex])
        return normValue
    
    def normalizeTest(feaIndex, OriValue):
        normValue = (OriValue - Knn.listTestMin[feaIndex]) / (Knn.listTestMax[feaIndex] - Knn.listTestMin[feaIndex])
        return normValue
    
    def randomSelectIntroBinary(self):
        train_total = list(Knn.listTrainCustomer)
        # traindata length for intro binary: # 200
        train_model, test_model = [],[]
        train_model = np.random.choice(train_total,int(128), replace=False)
        for row in train_total:
            if row not in train_model: test_model.append(row)
            
        #print("1 before",Knn.listTrainCustomer[0])
        Knn.listTrainCustomer = train_model
        #print("2 after", Knn.listTrainCustomer[0])
        Knn.listTestCustomer = test_model
        print("test data %:", int(len(train_model)/len(train_total)*100))
        
def cross_validation(k, n, train_file_name):
    # Knn = Knn()
    # Knn.readFromFile_Train(train_file_name)
    #
    # Knn.readFromFile_Test(test_file_name)
    #
    # Knn.randomSelectIntroBinary()
    #Knn.randomSelect()

    knn = Knn()
    # Prepare the train_set_total
    whole_set = knn.readFromFile_Train(train_file_name)[:]
    # print(whole_set)
    shuffle(whole_set)
    print(len(whole_set))
    test_size = int(len(whole_set) / n)
    print("test size " + str(test_size))

    performance = []
    for i in range(1, n + 1):
        # prepare the test set
        Knn.listTestCustomer = whole_set[test_size * (i - 1):test_size * i]
        # prepare the train set
        Knn.listTrainCustomer = []
        for j in range(0, len(whole_set)):
            if j < test_size * (i - 1) or j >= test_size * i:
                Knn.listTrainCustomer.append(whole_set[j])

        # record the min and max of test and train
        Knn.initializeTrain()
        Knn.initializeTest()

        for x in range(len(Knn.listTrainCustomer)):
            Knn.recordTrainMinMax(2, Knn.listTrainCustomer[x].mFee)
            Knn.recordTrainMinMax(3, Knn.listTrainCustomer[x].budget)
            Knn.recordTrainMinMax(6, Knn.listTrainCustomer[x].interest)
            Knn.recordTrainMinMax(7, Knn.listTrainCustomer[x].period)

        for x in range(len(Knn.listTestCustomer)):
            Knn.recordTestMinMax(2, Knn.listTestCustomer[x].mFee)
            Knn.recordTestMinMax(3, Knn.listTestCustomer[x].budget)
            Knn.recordTestMinMax(6, Knn.listTestCustomer[x].interest)
            Knn.recordTestMinMax(7, Knn.listTestCustomer[x].period)

        #print("train min",Knn.listTrainMin,"train max",Knn.listTrainMax,"test min",Knn.listTestMin,"test max", Knn.listTestMax)

        # normalization
        tmp_listA = []

        for row in Knn.listTrainCustomer:
            val0 = row.ser
            val1 = row.cus
            val2 = Knn.normalizeTrain(2, row.mFee)
            val3 = Knn.normalizeTrain(3, row.budget)
            val4 = row.size
            val5 = row.promo
            val6 = Knn.normalizeTrain(6, row.interest)
            val7 = Knn.normalizeTrain(7, row.period)
            val8 = row.label
            tmp_listA.append(Product(val0, val1, val2, val3, val4, val5, val6, val7, val8))

        Knn.listTrainCustomer = list(tmp_listA)


        tmp_listB = []

        for row in Knn.listTestCustomer:
            val2 = Knn.normalizeTest(2, row.mFee)
            val3 = Knn.normalizeTest(3, row.budget)
            val6 = Knn.normalizeTest(6, row.interest)
            val7 = Knn.normalizeTest(7, row.period)
            val8 = row.label
            tmp_listB.append(Product(row.ser, row.cus, val2, val3, row.size, row.promo, val6, val7, val8))


        Knn.listTestCustomer = list(tmp_listB)


        #print('train 19:',Knn.listTrainCustomer[19])

        ########################################################
        #  simlarity
        ########################################################
        # transfer the discrete to simlarity distance: Ctype, lifestyle

        w1, w2, w3, w4, w5, w6, w7, w8 = 1,1,1,1,1,1,1,1
        near = []
        # the index of test data

        #print(Knn.listTrainCustomer[t])

        for t in range(len(Knn.listTestCustomer)):
            for m in range(len(Knn.listTrainCustomer)):
            # ser, cus, mFee, budget, size, promo, interest, period
                sim_ser = Product.serTypeDistance[Knn.listTestCustomer[t].ser][Knn.listTrainCustomer[m].ser]
                sim_cus = Product.cusTypeDistance[Knn.listTestCustomer[t].cus][Knn.listTrainCustomer[m].cus]
                sim_size = Product.sizeDistance[Knn.listTestCustomer[t].size][Knn.listTrainCustomer[m].size]
                sim_promo = Product.promoDistance[Knn.listTestCustomer[t].promo][Knn.listTrainCustomer[m].promo]

            # get the overall simlarity distance
                dist_mFee = (Knn.listTrainCustomer[m].mFee - Knn.listTestCustomer[t].mFee)**2
                dist_budget = (Knn.listTrainCustomer[m].budget - Knn.listTestCustomer[t].budget)**2
                dist_interest = (Knn.listTrainCustomer[m].interest - Knn.listTestCustomer[t].interest)**2
                dist_period = (Knn.listTrainCustomer[m].period - Knn.listTestCustomer[t].period)**2
                if (dist_mFee==dist_budget==dist_interest==dist_period==0):
                    Knn.listTestCustomer[t].predL = Knn.listTrainCustomer[m].label

                else:
                    sim_overall = 1/((w1*(1-sim_ser) + w2*(1-sim_cus) + w3* dist_mFee+ \
                                     w4* dist_budget+ w5*(1-sim_size) + w6* (1-sim_promo) + w7*dist_interest + w8 * dist_period)**0.5)
                    a = Knn.listTrainCustomer.copy()
                    a[m].sim = sim_overall
                #     for row in a:
                #         print(row)
                    sor = sorted(a, key=lambda Product:Product.sim, reverse=True)
                    # get the nearest 3 distance
                    near = sor[0:k]

                    countSuccess = 0;
                    countFailure = 0
                    for e in range(0,k):
                        if near[e].label == 1.0:
                            countSuccess += 1;
                        elif near[e].label == 0.0:
                            countFailure += 1;
                    if countSuccess > countFailure:
                        Knn.listTestCustomer[t].predL = 1.0
                    else:
                        Knn.listTestCustomer[t].predL = 0.0

                    #print(Knn.listTestCustomer[t])

        #for row in Knn.listTrainCustomer:
            #print("TRAIN",row.label)
        # for row in knn.listTestCustomer:
            #print("TESTOri", row.label)
            #print("TEST",row.predL)
        count = 0;
        for row in Knn.listTestCustomer:
            if row.label == row.predL:
                count += 1

        print("count", count)
        percentage = count * 1.0 /len(Knn.listTestCustomer)
        print('percentage is')
        print(percentage)
        performance.append(percentage)
    print (performance)
    print (sum(performance) / n)

# main
k = int(sys.argv[1])
n = int(sys.argv[2])
train_file_name = sys.argv[3]
cross_validation(k, n, train_file_name)
cross_validation(k, n, train_file_name)
cross_validation(k, n, train_file_name)

