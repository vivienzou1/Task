import arff
import random
########################################################
# Data Loader: train data and test data split
########################################################
# def dataloader():
train = arff.loads(open('trainProdSelection.arff'))
train_total = (train.get('data'))
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
for i in range(25):
    train_C1.append(random.choice(train_total[0:35]))
    train_C5.append(random.choice(train_total[-36:-1]))
for i in range(18):
    train_C2.append(random.choice(train_total[36:61]))
for i in range(28):
    train_C3.append(random.choice(train_total[62:102]))
for i in range(32):
    train_C4.append(random.choice(train_total[103:149]))

train_model = train_C1 + train_C2 + train_C3 + train_C4 + train_C5

'''
create the test set that are included in the train_total but train_data
'''
for row in train_total:
    if row not in train_model: test_model.append(row)

print("test data %:", int(len(test_model)/186*100))

# return train_model, test_model
