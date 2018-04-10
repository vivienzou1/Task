
# coding: utf-8

# In[10]:


import numpy as np

class Customer:
    'common class for all types of customer'
    
    def __init__(self, service, customer, fee, advertisement,size, promotion, interest, period, label):  
        self.service = service
        self.customer = customer
        self.fee = fee
        self.advertisement = advertisement
        self.size = size
        self.promotion = promotion
        self.interest = interest
        self.period = period
        self.predC = 'not yet'
        self.sim = float(0)
        self.label = label
        
    def set_variable(self, value1, value2, value3, value4):
        self.fee = value1
        self.advertisement = value2
        self.interest = value3
        self.period = value4

    def __str__(self):
        return "serivce %s, customer %s, fee %f advertisement %f size %s promotion %s interest %f period %f label %s predicted %s sim %f" %    (self.service, self.customer, self.fee, self.advertisement, self.size, self.promotion, self.interest,      self.period, self.label, self.predC, self.sim)


# In[11]:


class Knn:
    'main class to operate all'
    listTrainMin = [None]*8
    listTrainMax = [None]*8
    listTestMin = [None]*8
    listTestMax = [None]*8
    listTrainCustomer = []
    listTestCustomer = []
        
    def readFromFile_Train(name):
        with open(name, "r") as f:         
            for line in f.readlines():
                if not line.startswith("@") and ',' in line:
                    array = line.split(',')   
                    service = array[0]
                    customer = array[1]
                    fee = float(array[2])
                    advertisement = float(array[3])
                    size = str(array[4])
                    promotion = str(array[5])
                    interest = float(array[6].strip())
                    period = float(array[7].strip())
                    label = str(array[8].strip())
                    Knn.listTrainCustomer.append(Customer(service, customer, fee, advertisement,size, promotion,                                                          interest, period, label))
#             for row in Knn.listTrainCustomer:
#                 print(row.fee) # PASS

    def readFromFile_Test(name):
        with open(name, "r") as f:         
            for line in f.readlines():
                if not line.startswith("@") and ',' in line:
                    array = line.split(',')   
                    service = array[0]
                    customer = array[1]
                    fee = float(array[2])
                    advertisement = float(array[3])
                    size = str(array[4])
                    promotion = str(array[5])
                    interest = float(array[6].strip())
                    period = float(array[7].strip())
                    label = 'unknown'
                    Knn.listTestCustomer.append(Customer(service, customer, fee, advertisement,size, promotion,                                                          interest, period, label))
#             for row in Knn.listTestCustomer:
#                 print(row) # PASS
                    
    def initializeTrain():
        for i in range(len(Knn.listTrainMin)):
            if i == 2 or i == 3 or i == 6 or i == 7:
                Knn.listTrainMin[i]= float('inf')
        for i in range(len(Knn.listTrainMax)):
            if i == 2 or i == 3 or i == 6 or i == 7:
                Knn.listTrainMax[i] = float('-inf')
            
    def initializeTest():
        for i in range(len(Knn.listTestMin)):
            if i == 2 or i == 3 or i == 6 or i == 7:
                Knn.listTestMin[i]= float('inf')
        for i in range(len(Knn.listTestMax)):
            if i == 2 or i == 3 or i == 6 or i == 7:
                Knn.listTestMax[i] = float('-inf')


    def randomSelect():
        #randomly select the 80/20
        train_total = list(Knn.listTrainCustomer)   
#         for row in Knn.listTrainCustomer:
#             print(row.fee)
        '''
        split 80% data randomly from each class individually as experiment data to build model
        return train_model, test_model
        '''
        each_fold = int(len(train_total)/8) # 20 PASS
        test_slice, train_slice = np.split(train_total.copy(), [each_fold], axis=0)
        '''
        RUN THE TEST BETWEEN EACH FOLD
        '''
#         train_slice[:each_fold], test_slice = test_slice, train_slice[:each_fold].copy()
#         train_slice[each_fold:2*each_fold], test_slice = test_slice, train_slice[each_fold:2*each_fold].copy()
#         train_slice[2*each_fold:3*each_fold], test_slice = test_slice, train_slice[2*each_fold:3*each_fold].copy()
#         train_slice[3*each_fold:4*each_fold], test_slice = test_slice, train_slice[3*each_fold:4*each_fold].copy()
#         train_slice[4*each_fold:5*each_fold], test_slice = test_slice, train_slice[4*each_fold:5*each_fold].copy()
#         train_slice[5*each_fold:6*each_fold], test_slice = test_slice, train_slice[5*each_fold:6*each_fold].copy()
#         train_slice[6*each_fold:7*each_fold], test_slice = test_slice, train_slice[6*each_fold:7*each_fold].copy()
#         train_slice[7*each_fold:8*each_fold], test_slice = test_slice, train_slice[7*each_fold:8*each_fold].copy()
        '''
        create the test set that are included in the train_total but train_data
        '''
        Knn.listTrainCustomer = train_slice
        Knn.listTestCustomer = test_slice
#         Knn.listRealTestCustomer = Knn.listTestCustomer
        
        for row in test_slice:
            print(row.fee) # PASS

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


# In[12]:


Knn.initializeTrain()
Knn.readFromFile_Train('trainProdIntro.binary.arff')

Knn.initializeTest()
Knn.readFromFile_Test('testProdIntro.binary.arff')

Knn.randomSelect()


# In[13]:


for i in range(len(Knn.listTrainCustomer)):
    Knn.recordTrainMinMax(2, Knn.listTrainCustomer[i].fee)
    Knn.recordTrainMinMax(3, Knn.listTrainCustomer[i].advertisement)
    Knn.recordTrainMinMax(6, Knn.listTrainCustomer[i].interest)
    Knn.recordTrainMinMax(7, Knn.listTrainCustomer[i].period)

for i in range(len(Knn.listTestCustomer)):
    Knn.recordTestMinMax(2, Knn.listTestCustomer[i].fee)
    Knn.recordTestMinMax(3, Knn.listTestCustomer[i].advertisement)
    Knn.recordTestMinMax(6, Knn.listTestCustomer[i].interest)   
    Knn.recordTestMinMax(7, Knn.listTestCustomer[i].period)


# In[14]:


# normalization
tmp_listA = []
for row in Knn.listTrainCustomer:
    val0 = row.service
    val1 = row.customer
    val2 = Knn.normalizeTrain(2, row.fee)
    val3 = Knn.normalizeTrain(3, row.advertisement)
    val4 = row.size
    val5 = row.promotion
    val6 = Knn.normalizeTrain(6, row.interest)
    val7 = Knn.normalizeTrain(7, row.period)
    val8 = row.label
    tmp_listA.append(Customer(val0, val1, val2, val3, val4, val5, val6, val7, val8))

Knn.listTrainCustomer = list(tmp_listA)

tmp_listB = [] 
for row in Knn.listTestCustomer:
    val0 = row.service
    val1 = row.customer
    val2 = Knn.normalizeTest(2, row.fee)
    val3 = Knn.normalizeTest(3, row.advertisement)
    val4 = row.size
    val5 = row.promotion
    val6 = Knn.normalizeTest(6, row.interest)
    val7 = Knn.normalizeTest(7, row.period)
    val8 = row.label
    tmp_listB.append(Customer(val0, val1, val2, val3, val4, val5, val6, val7, val8))
Knn.listTestCustomer = list(tmp_listB)


# In[15]:


for row in Knn.listTrainCustomer:
    if row.service == 'Loan': row.service = 0
    if row.service == 'Bank_Account':row.service = 1
    if row.service == 'CD':row.service = 2
    if row.service == 'Mortgage':row.service = 3
    if row.service == 'Fund':row.service = 4
    
    if row.customer == 'Business':row.customer = 0
    if row.customer == 'Professional':row.customer = 1
    if row.customer == 'Student':row.customer = 2
    if row.customer == 'Doctor':row.customer = 3
    if row.customer == 'Other':row.customer = 4
        
    if row.size == 'Small':row.size = 0
    if row.size == 'Medium':row.size = 1
    if row.size == 'Large':row.size = 2
    
    if row.promotion == 'Full':row.promotion = 0
    if row.promotion == 'Web&Email':row.promotion = 1
    if row.promotion == 'Web':row.promotion = 2
    if row.promotion == 'None':row.promotion = 3
        
for row in Knn.listTestCustomer:
    if row.service == 'Loan': row.service = 0
    if row.service == 'Bank_Account':row.service = 1
    if row.service == 'CD':row.service = 2
    if row.service == 'Mortgage':row.service = 3
    if row.service == 'Fund':row.service = 4
    
    if row.customer == 'Business':row.customer = 0
    if row.customer == 'Professional':row.customer = 1
    if row.customer == 'Student':row.customer = 2
    if row.customer == 'Doctor':row.customer = 3
    if row.customer == 'Other':row.customer = 4
        
    if row.size == 'Small':row.size = 0
    if row.size == 'Medium':row.size = 1
    if row.size == 'Large':row.size = 2
    
    if row.promotion == 'Full':row.promotion = 0
    if row.promotion == 'Web&Email':row.promotion = 1
    if row.promotion == 'Web':row.promotion = 2
    if row.promotion == 'None':row.promotion = 3


# In[16]:


# similarity for discrete
near = []
# for t in range(len(Knn.listTestCustomer)):
for t in range(0,1):
    for j in range(len(Knn.listTrainCustomer)):
        ser_matrix = [[1,0,0.1,.3,0.2],[0,1,0,0,0],[0.1,0,1,0.2,0.2],[0.3,0,0.2,1,0.1],[0.2,0,0.2,0.1,1]]
        cus_matrix = [[1,0.2,0.1,0.2,0],[0.2,1,0.2,0.1,0],[0.1,0.2,1,0.1,0],[0.2,0.1,0.1,1,0],[0,0,0,0,1]]
        size_matrix = [[1,0.1,0],[0.1,1,0.1],[0,0.1,1]]
        prom_matrix = [[1,0.8,0,0],[0.8,1,0.1,0.5],[0,0.1,1,0.4],[0,0.5,0.4,1]]
        sim_ser = ser_matrix[Knn.listTestCustomer[t].service][Knn.listTrainCustomer[j].service]
        sim_cus = cus_matrix[Knn.listTestCustomer[t].customer][Knn.listTrainCustomer[j].customer]
        sim_size = size_matrix[Knn.listTestCustomer[t].size][Knn.listTrainCustomer[j].size]
        sim_prom = prom_matrix[Knn.listTestCustomer[t].promotion][Knn.listTrainCustomer[j].promotion]
        
        # get the overall simlarity distance
        dist_fee = (Knn.listTrainCustomer[j].fee - Knn.listTestCustomer[t].fee)**2
        dis_advertisement = (Knn.listTrainCustomer[j].advertisement - Knn.listTestCustomer[t].advertisement)**2
        dist_interest = (Knn.listTrainCustomer[j].interest - Knn.listTestCustomer[t].interest)**2
        dist_promotion = (Knn.listTrainCustomer[j].promotion - Knn.listTestCustomer[t].promotion)**2
        
        # calculating overall similarity
        if (dist_fee==dis_advertisement==dist_interest==dist_promotion==0):
            Knn.listTestCustomer[t].predC = Knn.listTrainCustomer[j].label

        else:
            # default weight
#             w1,w2,w3,w4,w5,w6,w7,w8 = 0.1,0,0.2,3,0.4,2.5,1,1
            w1,w2,w3,w4,w5,w6,w7,w8 = 1,1,1,1,1,1,1,1
#             w1,w2,w3,w4,w5,w6,w7,w8 = 38 , 0.03347663665063675 , 4 , 92 , 22 , 83, 1, 1
            sim_overall = 1/((w1*(1-sim_ser) + w2*(1-sim_cus) + w5*(1-sim_size) + w6*(1-sim_prom)                               + w3* dist_fee+ w4* dis_advertisement+ w7*dist_interest + w8* dist_promotion)                             **0.5)
            a = Knn.listTrainCustomer.copy()
            a[j].sim = sim_overall
            sor = sorted(a, key=lambda Customer:Customer.sim, reverse=True)
#             for row in sor:
#                 print("lable",row)
#             print()
            # k = 5
            near = sor[0:5]
            for row in near:
                print("lable",row)
            #################### NEW #################
            C1, C0 = 0, 0
            for n in range(0,5):
                if near[n].label == '1': 
#                     print(near[n].label)
                    C1 += 1
                if near[n].label == '0': 
#                     print(near[n].label)
                    C0 += 1
            print(C1,C0)
#             highest = max(C1, C0)
#             print(highest)
            if C1 > C0:
                Knn.listTestCustomer[t].predC = '1'
                print("label",Knn.listTestCustomer[t].label)
                print("pred",Knn.listTestCustomer[t].predC)
                print("t",t)
            elif C1< C0:
                Knn.listTestCustomer[t].predC = '0'
                print("label",Knn.listTestCustomer[t].label)
                print("pred",Knn.listTestCustomer[t].predC)
                print("t",t)
               


# In[17]:


for row in Knn.listTestCustomer:
    print(row)


# In[18]:


count = 0;
for row in Knn.listTestCustomer:
#     print(row)
#     print("pred:",row.predC)
#     print("label:",row.label)
    if row.predC == row.label:
        count += 1
print("count", count)
percentage = count/len(Knn.listTestCustomer)*100
print('percentage is',percentage)


# In[10]:


# 

