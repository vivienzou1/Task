
# coding: utf-8

# In[172]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 16:29:52 2018

@author: yinchenli
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 13:37:49 2018
@author: yinchenli
"""
import numpy as np

class Customer:
    'common class for all types of customer'
    
    def __init__(self, Ctype,lifestyle,vacation,eCredit,salary,prop,cla):  
        self.Ctype = Ctype
        self.lifestyle = lifestyle
        self.vacation = vacation
        self.eCredit = eCredit
        self.salary = salary
        self.prop = prop
        self.predC = 'not predicted'
        self.sim = float(0)
        self.cla = cla

    def set_variable(self, value1, value2, value3, value4):
        self.vacation = value1
        self.eCredit = value2
        self.salary = value3
        self.prop = value4

    def __str__(self):
        return "this customer: type %s, life %s, vac %f eCredit %f salary %f prop %f class %s predicted class %s sim %f" %     (self.Ctype, self.lifestyle, self.vacation, self.eCredit, self.salary, self.prop, self.cla, self.predC, self.sim)

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
                    array = line.split(',')   
                    Ctype = array[0]
                    lifestyle = array[1]
                    vacation = float(array[2])
                    eCredit = float(array[3])
                    salary = float(array[4])
                    prop = float(array[5].strip())
                    cla = str(array[6].strip())
                    Knn.listTrainCustomer.append(Customer(Ctype,lifestyle,vacation,eCredit,salary,prop,cla))
    
    
    def readFromFile_Test(name):
        with open(name, "r") as f:         
            for line in f.readlines():
                if not line.startswith("@") and ',' in line:
                    array = line.split(',')   
                    Ctype = array[0]
                    lifestyle = array[1]
                    vacation = float(array[2])
                    eCredit = float(array[3])
                    salary = float(array[4])
                    prop = float(array[5].strip())
                    cla = 'unknown'
                    Knn.listTestCustomer.append(Customer(Ctype,lifestyle,vacation,eCredit,salary,prop,cla))
                                        
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
    
    def randomSelect():
        #randomly select the 80/20
        train_total = list(Knn.listTrainCustomer)     
        # print(len(train_data)) # 186
        # length with C1 label: 36
        # length with C2 label: 26
        # length with C3 label: 41
        # length with C4 label: 47
        # length with C5 label: 36
        # print(len(train_model)) # 128
        '''
        split 70% data randomly from each class individually as experiment data to build model
        def data_loader():
        return train_model, test_model
        '''
        train_model, test_model, train_C1, train_C2, train_C3, train_C4, train_C5 =[], [], [], [], [], [], []
        copy_train1 = list(train_total[0:36])
        copy_train2 = list(train_total[36:61])
        copy_train3 = list(train_total[62:102])
        copy_train4 = list(train_total[103:149])
        copy_train5 = list(train_total[-36:-1])
        train_C1 = np.random.choice(copy_train1,int(29))

        train_C2 = np.random.choice(copy_train2,int(21))

        train_C3 = np.random.choice(copy_train3,int(34))
        train_C4 = np.random.choice(copy_train4,int(38))
        train_C5 = np.random.choice(copy_train5,int(29))

        train_model = list(train_C1) + list(train_C2) + list(train_C3) + list(train_C4) + list(train_C5)
        
        '''
        create the test set that are included in the train_total but train_data
        '''
        for row in train_total:
            if row not in train_model: test_model.append(row)
        
        #print("1 before",Knn.listTrainCustomer[0])
        Knn.listTrainCustomer = train_model
        #print("2 after", Knn.listTrainCustomer[0])
        Knn.listTestCustomer = test_model
        
        print("test data %:", int(len(train_C1)/36*100))

Knn.initializeTrain()
Knn.readFromFile_Train('trainProdSelection.arff')

Knn.initializeTest()
Knn.readFromFile_Test('testProdSelection.arff')

Knn.randomSelect()
# record the min and max of test and train

for i in range(len(Knn.listTrainCustomer)):
    Knn.recordTrainMinMax(2, Knn.listTrainCustomer[i].vacation)
    Knn.recordTrainMinMax(3, Knn.listTrainCustomer[i].eCredit)
    Knn.recordTrainMinMax(4, Knn.listTrainCustomer[i].salary)
    Knn.recordTrainMinMax(5, Knn.listTrainCustomer[i].prop)

for i in range(len(Knn.listTestCustomer)):
    Knn.recordTestMinMax(2, Knn.listTestCustomer[i].vacation)
    Knn.recordTestMinMax(3, Knn.listTestCustomer[i].eCredit)
    Knn.recordTestMinMax(4, Knn.listTestCustomer[i].salary)   
    Knn.recordTestMinMax(5, Knn.listTestCustomer[i].prop)

# print("train min",Knn.listTrainMin,"train max",Knn.listTrainMax,"test min",Knn.listTestMin,"test max", Knn.listTestMax)

# normalization
tmp_listA = []
for row in Knn.listTrainCustomer:
    val0 = row.Ctype
    val1 = row.lifestyle
    val2 = Knn.normalizeTrain(2, row.vacation)
    val3 = Knn.normalizeTrain(3, row.eCredit)
    val4 = Knn.normalizeTrain(4, row.salary)
    val5 = Knn.normalizeTrain(5, row.prop)
    val6 = row.cla
    tmp_listA.append(Customer(val0, val1, val2, val3, val4, val5, val6))
    
Knn.listTrainCustomer = list(tmp_listA)
# for row in Knn.listTrainCustomer:
#     print (row)

tmp_listB = [] 
for row in Knn.listTestCustomer:
    val1 = row.lifestyle
    val2 = Knn.normalizeTrain(2, row.vacation)
    val3 = Knn.normalizeTrain(3, row.eCredit)
    val4 = Knn.normalizeTrain(4, row.salary)
    val5 = Knn.normalizeTrain(5, row.prop)
    val6 = row.cla
    tmp_listB.append(Customer(row.Ctype, val1, val2, val3, val4, val5, val6))
Knn.listTestCustomer = list(tmp_listB)
########################################################
#  simlarity 
########################################################
# transfer the discrete to simlarity distance: Ctype, lifestyle
w1, w2, w3, w4, w5, w6 = 1,0.001,1,25,4,25
near = []
# the index of test data

#print(Knn.listTrainCustomer[t])
for t in range(len(Knn.listTestCustomer)):
    for j in range(len(Knn.listTrainCustomer)):

        if Knn.listTestCustomer[t].Ctype ==(Knn.listTrainCustomer[j].Ctype):
            sim_type = 1
        else:
            sim_type = 0
        
        if Knn.listTestCustomer[t].lifestyle ==(Knn.listTrainCustomer[j].lifestyle):
            sim_ls = 1
        else:
            sim_ls = 0

# get the overall simlarity distance
        dist_vaca = (Knn.listTrainCustomer[j].vacation - Knn.listTestCustomer[t].vacation)**2
        dis_eCred = (Knn.listTrainCustomer[j].eCredit - Knn.listTestCustomer[t].eCredit)**2
        dist_salary = (Knn.listTrainCustomer[j].salary - Knn.listTestCustomer[t].salary)**2
        dist_prop = (Knn.listTrainCustomer[j].prop - Knn.listTestCustomer[t].prop)**2
        if (dist_vaca==dis_eCred==dist_salary==dist_prop==0):
            Knn.listTestCustomer[t].predC = Knn.listTrainCustomer[j].cla

        else:
            sim_overall = 1/((w1*(1-sim_type) + w2*(1-sim_ls) + w3* dist_vaca+                          w4* dis_eCred+ w5*dist_salary + w6* dist_prop)**0.5)
            a = Knn.listTrainCustomer.copy()
            a[j].sim = sim_overall
            sor = sorted(a, key=lambda Customer:Customer.sim, reverse=True)
            near = sor[0:3]
            C1, C2, C3, C4, C5 = 0,0,0,0,0
            for n in range(0,3):
                if near[n].cla == 'C1': 
                    C1 += 1
                if near[n].cla == 'C2': 
                    C2 += 1
                if near[n].cla == 'C3': 
                    C3 += 1
                if near[n].cla == 'C4': 
                    C4 += 1
                if near[n].cla == 'C5': 
                    C5 += 1     
            #print(C1, C2, C3, C4, C5)
            highest = max(C1, C2, C3, C4, C5)
            if highest == C1:
                Knn.listTestCustomer[t].predC = 'C1'
            elif highest == C2:
                Knn.listTestCustomer[t].predC = 'C2'
            elif highest == C3:
                Knn.listTestCustomer[t].predC = 'C3'
            elif highest == C4:
                Knn.listTestCustomer[t].predC = 'C4'
            elif highest == C5:
                Knn.listTestCustomer[t].predC = 'C5'

count = 0;
for row in Knn.listTestCustomer:
    if row.predC == row.cla:
        count += 1

print("count", count)
percentage = count/len(Knn.listTestCustomer)*100
print('percentage is')
print(percentage)
    

