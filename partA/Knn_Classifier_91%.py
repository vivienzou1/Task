
# coding: utf-8

# In[18]:


import numpy as np


# # setting up the Customer type

# In[19]:


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


# # the model

# In[20]:


def model(w1, w2, w3, w4, w5, w6):
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

            Knn.listTrainCustomer = train_model
            # the list of test code lacked of label
            Knn.listRealTestCustomer = Knn.listTestCustomer
            # the list of cross validation test data with label already
#             Knn.listTestCustomer = test_model



    Knn.initializeTrain()
    Knn.readFromFile_Train('trainProdSelection.arff')

    Knn.initializeTest()
    Knn.readFromFile_Test('testProdSelection.arff')

    Knn.randomSelect()
    
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
        
    for i in range(len(Knn.listRealTestCustomer)):
        Knn.recordTestMinMax(2, Knn.listRealTestCustomer[i].vacation)
        Knn.recordTestMinMax(3, Knn.listRealTestCustomer[i].eCredit)
        Knn.recordTestMinMax(4, Knn.listRealTestCustomer[i].salary)   
        Knn.recordTestMinMax(5, Knn.listRealTestCustomer[i].prop)

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
                sim_overall = 1/((w1*(1-sim_type) + w2*(1-sim_ls) + w3* dist_vaca+                              w4* dis_eCred+ w5*dist_salary + w6* dist_prop)**0.5)
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
#     for row in Knn.listTestCustomer:
#         if row.predC == row.cla:
#             count += 1

#     print("count", count)
#     percentage = count/len(Knn.listTestCustomer)*100
#     print('percentage is',percentage)
#     return percentage,w1, w2, w3, w4, w5, w6
    return Knn.listTestCustomer


# # To Train the data

# In[21]:


def train():
    accuracyAcc = 0
    epoch = 200
    max_percentage = 0.0
    best_w1, best_w2, best_w3, best_w4, best_w5, best_w6 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for i in range(epoch):
        w1 = int(np.random.randint(low=0, high=50, size=1))
        w2 = np.random.random_sample()
        w3 = int(np.random.randint(low=0, high=50, size=1))
        w4 = int(np.random.randint(low=20, high=150, size=1))
        w5 = int(np.random.randint(low=0, high=50, size=1))
        w6 = int(np.random.randint(low=20, high=100, size=1))

        w4 = np.random.uniform(0,100)
        w2 = np.random.random_sample()
        w3 = np.random.uniform(0,100) 
        w4 = np.random.uniform(20,1000)
        w5 = np.random.uniform(0,400) 
        w6 = np.random.uniform(20,500) 

        each_percentage, guess_w1, guess_w2, guess_w3, guess_w4, guess_w5, guess_w6 = model(w1, w2, w3, w4, w5, w6)
        accuracyAcc += each_percentage

        if each_percentage > max_percentage:
            max_percentage = each_percentage
            best_w1,best_w2, best_w3, best_w4, best_w5, best_w6 = guess_w1, guess_w2,             guess_w3, guess_w4, guess_w5, guess_w6      

    averageAcc = accuracyAcc/epoch
    print("average accuracy: ",averageAcc)
    print("max percentage: ",max_percentage)
    print("best weights: ",best_w1,", ",best_w2,", ", best_w3,", ", best_w4,", ", best_w5,", ", best_w6)


# In[22]:


def write_results(classification_result, output_file='predictions.txt'):
    with open(output_file, 'w') as f:
        for row in classification_result:
            f.write('\n'+str(row))


# In[23]:


def main():
    # if under train mode:
    # train()

    w1, w2, w3, w4, w5, w6 = 38 , 0.03347663665063675 , 4 , 92 , 22 , 83
    classification_result = model(w1, w2, w3, w4, w5, w6)

    write_results(classification_result)


# In[24]:


main()


# #### average accuracy:  77.44215255474278
# #### best weights:  65.71608485534179 ,  1 ,  1 ,  1 ,  1 ,  1

# #### average accuracy:  77.57041651904352
# #### best weights:  65.71608485534179 ,  86.85199055850295 ,  1 ,  1 ,  1 ,  1

# #### average accuracy:  74.73992247757633
# #### best weights:  65.71608485534179 ,  86.85199055850295 ,  69.02819705097116 ,  1 ,  1 ,  1

# # globally random

# #### max percentage:  97.46835443037975
# #### best weights:  1 ,  0.001 ,  1 ,  25 ,  4 ,  25
# #### average accuracy:  90.28926636099446

# #### max percentage:  96.55172413793103
# #### best weights:  6 ,  0.09185761914944286 ,  9 ,  78 ,  20 ,  83
# #### average accuracy:  90.5521689254

# #### max percentage:  96.42857142857143
# #### best weights:  38 ,  0.03347663665063675 ,  4 ,  92 ,  22 ,  83
# #### average accuracy:  90.6405599389135 
# #### i= 200

# #### max percentage:  98.7012987012987
# #### best weights:  44 ,  0.3073969273139102 ,  11 ,  76 ,  25 ,  73
# #### average accuracy:  90.46584300975651

# In[ ]:


#### max percentage:  97.67441860465115
#### best weights:  3.931178960471504 ,  0.32915796147046716 ,  12.57601098249561 ,  130.31784814976396 ,  17.122452707211767 ,  93.3587889389484
#### average accuracy:  90.30189873494271


# In[ ]:


#### max percentage:  96.47058823529412
#### best weights:  20.461545317666808 ,  0.33720149482716 ,  80.83690041034198 ,  702.2538324496462 ,  254.96421049469294 ,  418.21058515075964
#### average accuracy:  90.25981377813658

