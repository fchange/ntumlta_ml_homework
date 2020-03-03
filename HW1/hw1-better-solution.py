import csv
import sys

import numpy as np

# run as : python hw1-better-solution.py ./Dataset/train.csv
# 文件拷贝自 hw1.py
# 将training data 分成两部分，一部分 train，一部分验证

# read training data
raw_data = np.genfromtxt(sys.argv[1], delimiter=',')
data = raw_data[1:, 3:]
NansIndex = np.isnan(data)
data[NansIndex] = 0

month_to_data = {}  ## Dictionary (key:month, value:data)

for month in range(12):
    sample = np.empty(shape=(18, 20 * 24))
    for day in range(20):
        for hour in range(24):
            sample[:, day * 24 + hour] = data[18 * (month * 20 + day):18 * (month * 20 + day + 1), hour]
    month_to_data[month] = sample

x_full_data = np.empty(shape=(12 * 471, 18 * 9), dtype=float)
y = np.empty(shape=(12 * 471, 1), dtype=float)
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour > 14:
                continue
            x_full_data[month * 471 + day * 24 + hour, :] = month_to_data[month][:,
                                                            day * 24 + hour:day * 24 + hour + 9].reshape(1, -1)
            y[month * 471 + day * 24 + hour, 0] = month_to_data[month][9, day * 24 + hour + 9]

## take all
x = x_full_data

## only take PM10、PM2.5、SO2
# x = np.empty(shape=(12 * 471, 3 * 9), dtype=float)
# for i in range(9):
#     x[:, 3 * i + 0] = x_full_data[:, 9 * i + 8]
#     x[:, 3 * i + 1] = x_full_data[:, 9 * i + 9]
#     x[:, 3 * i + 2] = x_full_data[:, 9 * i + 12]

## only take PM2.5
# x = np.empty(shape=(12 * 471, 9), dtype=float)
# for i in range(9):
#     x[:, i] = x_full_data[:, 9 * i + 9]

## normalize
mean = np.mean(x, axis=0)
std = np.std(x, axis=0)
for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        if not std[j] == 0:
            x[i][j] = (x[i][j] - mean[j]) / std[j]

verificationLength = int(x.shape[0] / 5)
x_more = x[0:verificationLength:, :]
y_more = y[0:verificationLength:, :]
x = x[verificationLength+1:, :]
y = y[verificationLength+1:, :]

# Training
dim = x.shape[1] + 1
x = np.concatenate((x, np.ones(shape=(x.shape[0], 1))), axis=1).astype(float)
w = np.zeros(shape=(dim, 1))
learning_rate = np.array([[0.0000001]] * dim)
# learning_rate = np.array([[200]] * dim) # for adagrad
# adagrad_sum = np.zeros(shape=(dim, 1)) # for adagrad

for T in range(10000):
    Loss = y - x.dot(w)
    gradient = (-2) * np.transpose(x).dot(Loss)
    w = w - learning_rate * gradient
    # adagrad_sum += gradient ** 2
    # w = w - learning_rate * gradient / np.sqrt(adagrad_sum)
    if T % 500 == 0:
        print("T= ", T)
        print("L= ", np.power(np.sum(np.power(Loss, 2)) / x.shape[0], 0.5))
np.save('weight.npy', w)

# Verification
w = np.load('weight.npy')   ## load weight
x_more = np.concatenate((x_more, np.ones(shape=(x_more.shape[0], 1))), axis=1).astype(float)
Loss = y_more - x_more.dot(w)
print("===============================")
print("L= ", np.power(np.sum(np.power(Loss, 2)) / x_more.shape[0], 0.5))

# 拿前五分之一的数据做检验，成绩还不错，挺“拟合”的



