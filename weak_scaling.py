P = 1

# N = 10000000 * P
# R = 300
# K = 10000

N = 10 * P
R = 3
K = 3

tot_pesi_P_1 = 1
for t in range(1, K):  # t inizia da 1 e arriva fino a N
    print(t) 
    tot_pesi_P_1 += (N - t * (R - 1)) * R

print(tot_pesi_P_1)

P = 14

tot_pesi_P_10 = 0
for t in range(1, K):  # t inizia da 1 e arriva fino a N
    # print(t) 
    tot_pesi_P_10 += (N - t * (R - 1)) * R

print(tot_pesi_P_10)

print(tot_pesi_P_10/tot_pesi_P_1 * P)