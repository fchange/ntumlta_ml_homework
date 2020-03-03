import csv
import sys

import numpy as np

# run as : python hw1.py ./Dataset/train.csv ./Dataset/test.csv ./Dataset/ans.csv
# 真是一个差劲的模型
# 答案都差强人意

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


# Read in testing set
w = np.load('weight.npy')                                   ## load weight
test_raw_data = np.genfromtxt(sys.argv[2], delimiter=',')   ## test.csv
test_data = test_raw_data[:, 2: ]
NansIndex = np.isnan(test_data)
test_data[NansIndex] = 0

# Predict
## take all
test_x = np.empty(shape=(240, 18 * 9))
for i in range(240):
    test_x[i, :] = test_data[18*i:18*(i+1), :].reshape(1, 18 * 9)
## only take PM2.5
# test_x = np.empty(shape=(240, 9), dtype=float)
# for i in range(240):
#     test_x[i, :] = test_data[18 * i + 9, :]

## Normalization
mean = np.mean(test_x, axis=0)
std = np.std(test_x, axis=0)
for i in range(test_x.shape[0]):
    for j in range(test_x.shape[1]):
        if not std[j] == 0:
            test_x[i][j] = (test_x[i][j] - mean[j]) / std[j]

test_x = np.concatenate((np.ones(shape=(test_x.shape[0], 1)), test_x), axis=1).astype(float)
answer = test_x.dot(w)

## Write File
f = open(sys.argv[3], 'w')
w = csv.writer(f)
w.writerow(['id', 'value'])
for i in range(answer.shape[0]):
    w.writerow(['id_' + str(i), int(answer[i][0])])


