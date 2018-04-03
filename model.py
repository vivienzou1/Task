########################################################
#  simlarity
########################################################
# transfer the discrete to simlarity distance: Ctype, lifestyle
w1, w2, w3, w4, w5, w6 = 1,1,1,1,1,1
near = []
# the index of test data
t = 19
print(Knn.listTestCustomer[t])
for j in range(len(Knn.listTrainCustomer)):
# for j in range(3):
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
    sim_overall = 1/((w1*(1-sim_type) + w2*(1-sim_ls) + w3* dist_vaca+ \
                     w4* dis_eCred+ w5*dist_salary + w6* dist_prop)**0.5)
    a = Knn.listTrainCustomer.copy()
    a[j].sim = sim_overall
#     for row in a:
#         print(row)
    sor = sorted(a, key=lambda Customer:Customer.sim, reverse=False)
    # get the nearest 3 distance
    near = sor[0:3]
#     for row in near:
#         print(row)
#     print(near[0].cla)
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
#     print(C1, C2, C3, C4, C5)


print(Knn.listTestCustomer[0])
