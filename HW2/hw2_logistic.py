import numpy as np
import matplotlib.pyplot as plt


def _sigmoid(z):
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-6, 1 - 1e-6);


def get_prob(X, w, b):
    return _sigmoid(np.add(np.matmul(X, w), b))


def infer(X, w, b):
    return np.round(get_prob(X, w, b))


def _cross_entropy(y_pred, Y_label):
    return -np.dot(Y_label, np.log(y_pred)) - np.dot(1 - Y_label, np.log(1 - y_pred))


def _gradient(X, Y_label, w, b):
    y_pred = get_prob(X, w, b)
    pred_error = _cross_entropy(y_pred, Y_label)
    w_grad = -np.mean(np.multiply(pred_error.T, X.T), 1)
    b_grad = -np.mean(pred_error)
    return w_grad, b_grad


def _greadient_regularization(X, Y_label, w, b, lamda):
    y_pred = get_prob(X, w, b)
    pred_error = Y_label - y_pred
    w_grad = -np.mean(np.multiply(pred_error.T, X.T), 1) + lamda * w
    b_grad = -np.mean(pred_error)
    return w_grad, b_grad


def _loss(y_pred, Y_label, lamda, w):
    return _cross_entropy(y_pred, Y_label) + lamda * np.sum(np.square(w))


def accuracy(y_pred, Y_label):
    return np.sum(y_pred == Y_label) / Y_label


def _normalize_coloumn__normal(X, train=True, specified_Column=None, X_mean=None, Y_mean=None):
    if train:
        if specified_Column is None:
            specified_Column = np.arange(X.shape[1])
        length = len(specified_Column)
        X_mean = np.reshape(np.mean(X[:, specified_Column], 0), (1, length))
        X_std = np.reshape(np.std(X[:, specified_Column], 0), (1, length))

    X[:, specified_Column] = np.divide(np.subtract(X[:, specified_Column], X_mean), X_std)


def _shuffle(X, Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])


def train_dev_split(X, Y, dev_size=0.1155):
    train_len = int(round(len(X) * (1 - dev_size)))
    return X[:train_len], Y[:train_len], X[train_len + 1:], Y[train_len + 1:]


def train(X_train, Y_train):
    X_train, Y_train, X_dev, Y_dev = train_dev_split(X_train, Y_train)

    # Use 0 +0*x1 + 0*x2 + ... for weight initialization
    w = np.zeros((X_train.shape[1],))
    b = np.zeros((1,))

    regularize = True
    if regularize:
        lamda = 0.001
    else:
        lamda = 0

    max_iter = 40
    batch_size = 32
    learning_rate = 0.2
    num_train = len(Y_train)
    num_dev = len(Y_dev)
    step = 1

    loss_train = []
    loss_validation = []
    train_acc = []
    dev_acc = []

    for epoch in range(max_iter):
        # Random shuffle for each epoch
        X_train, Y_train = _shuffle(X_train, Y_train)

        total_loss = 0.0
        # logistic regression train with batch
        for idx in range(int(np.floor(len(Y_train) / batch_size))):
            X = X_train[idx * batch_size: (idx + 1) * batch_size]
            Y = Y_train[idx * batch_size: (idx + 1) * batch_size]

            w_grad, b_grad = _greadient_regularization(X, Y, w, b, lamda)

            w = w - learning_rate / np.sqrt(step) * w_grad
            b = b - learning_rate / np.sqrt(step) * b_grad

            step += 1

        y_train_pred = get_prob(X_train, w, b)
        Y_train_pred = infer(X_train, w, b)
        train_acc.append((accuracy(Y_train_pred, Y_train)))
        loss_train.append(_loss(y_train_pred, Y_train, lamda, w) / num_train)

        y_dev_pred = get_prob(X_dev, w, b)
        Y_dev_pred = np.round(y_dev_pred)
        dev_acc.append(accuracy(Y_dev_pred, Y_dev))
        loss_validation.append(_loss(y_dev_pred, Y_dev, lamda, w) / num_dev)

    return w, b, loss_train, loss_validation, train_acc, dev_acc  # return loss for plotting

X_train_fpath = 'data/X_train'
Y_train_fpath = 'data/Y_train'
X_test_fpath = 'data/X_test'
output_fpath = 'output.csv'

X_train = np.genfromtxt(X_train_fpath, delimiter=',', skip_header=1)
Y_train = np.genfromtxt(Y_train_fpath, delimiter=',', skip_header=1)
w, b, loss_train, loss_validation, train_acc, dev_acc= train(X_train, Y_train)

plt.title("Matplotlib demo")
plt.plot(loss_train)
plt.plot(loss_validation)
plt.legend(['train', 'dev'])
# plt.show()
plt.savefig('foo.png')
plt.show()

plt.plot(train_acc)
plt.plot(dev_acc)
plt.legend(['train', 'dev'])
# plt.show()
plt.savefig('foo1.png')
plt.show()