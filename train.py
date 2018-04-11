def train():
    accuracyAcc = 0
    epoch = 200
    max_percentage = 0.0
    best_w1, best_w2, best_w3, best_w4, best_w5, best_w6 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for i in range(epoch):
        # w1 = int(np.random.randint(low=0, high=50, size=1))
        # w2 = np.random.random_sample()
        # w3 = int(np.random.randint(low=0, high=50, size=1))
        # w4 = int(np.random.randint(low=20, high=150, size=1))
        # w5 = int(np.random.randint(low=0, high=50, size=1))
        # w6 = int(np.random.randint(low=20, high=100, size=1))

        w4 = np.random.uniform(0,1)
        w2 = np.random.random_sample()
        w3 = np.random.uniform(0,100) 
        w4 = np.random.uniform(20,1000)
        w5 = np.random.uniform(0,400) 
        w6 = np.random.uniform(20,500) 

        each_percentage, guess_w1, guess_w2, guess_w3, guess_w4, guess_w5, guess_w6 = model(w1, w2, w3, w4, w5, w6)
        accuracyAcc += each_percentage

        if each_percentage > max_percentage:
            max_percentage = each_percentage
            best_w1,best_w2, best_w3, best_w4, best_w5, best_w6 = guess_w1, guess_w2, guess_w3, guess_w4, guess_w5, guess_w6      

    averageAcc = accuracyAcc/epoch
    print("average accuracy: ",averageAcc)
    print("max percentage: ",max_percentage)
    print("best weights: ",best_w1,", ",best_w2,", ", best_w3,", ", best_w4,", ", best_w5,", ", best_w6)