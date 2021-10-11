import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def square_loss(y_pred_value, target):
    return np.mean(pow((y_pred_value - target), 2))


# def get_derivative(train_value, error):
#     return -2 * (train_value * error), -2 * error
#
#
# noise = np.random.normal(0, 1, 200)
# x = np.random.uniform(0, 3, 200)
#
# a = 2
# b = -4
#
# Y = a * x + b + noise
#
# plt.scatter(x, Y, marker='.', color='red')
# plt.plot()
# plt.plot(x, a * x + b)
# # plt.show()
#
# array = np.arange(200)
# np.random.shuffle(array)
#
# x_train = x[array]
# y_train = Y[array]
#
step = 0.01
# for _ in range(200):
#     test_y = a * x_train + b
#     e = y_train - test_y
#
#     a_deriv, b_deriv = get_derivative(x_train, e)
#
#     a = a - step * a_deriv.mean()
#     b = b - step * b_deriv.mean()
#
# print(a, b)

mean1 = [3, 4]
cov1 = [[1, 0], [0, 1]]
x1 = np.random.multivariate_normal(mean1, cov1, 200)
plt.scatter(x1[:, 0], x1[:, 1], marker='+', color='red')

mean2 = [5, 8]
cov2 = [[2, 0], [0, 2]]
x2 = np.random.multivariate_normal(mean2, cov2, 200)
plt.scatter(x2[:, 0], x2[:, 1], marker='*', color='blue')

x_combined = np.concatenate([x1, x2])
labels = np.array([0] * 200 + [1] * 200)

columns = ['X', 'Y']
df = pd.DataFrame(data=x_combined, columns=columns)

# df['lab'] = labels
#
# X = df.drop(['lab'], axis=1)
# Y = df['lab']
test = df.sample(80)
train = df[~df.isin(test)]
train.dropna(inplace=True)

X_train, y_train, X_test, y_test = train.X, train.Y, test.X, test.Y

W = np.random.uniform(0, 1)
b = 0.1

sigmoid = lambda x: 1 / (1 + np.exp(-x))


def fit(self, X, y, n_iterations=4):
    # Add dummy ones for bias weights
    X = np.insert(X, 0, 1, axis=1)

    n_samples, n_features = np.shape(X)

    # Initial parameters between [-1/sqrt(N), 1/sqrt(N)]

    # Tune parameters for n iterations
    for i in range(n_iterations):
        # Make a new prediction
        y_pred = sigmoid(X.dot(self.param))
        # Make a diagonal matrix of the sigmoid gradient column vector
        diag_gradient = make_diagonal(sigmoid_gradient(X.dot(self.param)))
        # Batch opt:
        self.param = np.linalg.pinv(X.T.dot(diag_gradient).dot(X)).dot(X.T).dot(
            diag_gradient.dot(X).dot(self.param) + y - y_pred)


def predict(self, X):
    # Add dummy ones for bias weights
    X = np.insert(X, 0, 1, axis=1)
    # Print a final prediction
    dot = X.dot(self.param)
    y_pred = np.round(sigmoid(dot)).astype(int)
    return y_pred