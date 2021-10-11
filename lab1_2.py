import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import fmin_tnc

sigmoid = lambda x: 1 / (1 + np.exp(-x))


def net_input(theta, x):
    # Computes the weighted sum of inputs Similar to Linear Regression

    return np.dot(x, theta)


def probability(theta, x):
    # Calculates the probability that an instance belongs to a particular class

    return sigmoid(net_input(theta, x))


def cost_function(theta, x, y):
    # Computes the cost function for all the training samples
    m = x.shape[0]
    total_cost = -(1 / m) * np.sum(
        y * np.log(probability(theta, x)) + (1 - y) * np.log(
            1 - probability(theta, x)))
    return total_cost


def gradient(theta, x, y):
    # Computes the gradient of the cost function at the point theta
    m = x.shape[0]
    return (1 / m) * np.dot(x.T, sigmoid(net_input(theta, x)) - y)

def fit(x, y, theta):
    opt_weights = fmin_tnc(func=cost_function, x0=theta, fprime=gradient,
                           args=(x, y))
    w = opt_weights[0]
    return w

def predict(w, x):
    theta = w[:, np.newaxis]
    return probability(theta, x)

def accuracy(self, x, actual_classes, probab_threshold=0.5):
    predicted_classes = (self.predict(x) >= probab_threshold).astype(int)
    predicted_classes = predicted_classes.flatten()
    accuracy = np.mean(predicted_classes == actual_classes)
    return accuracy * 100


mean1 = [3, 4]
cov1 = [[1, 0], [0, 1]]
x1 = np.random.multivariate_normal(mean1, cov1, 200)
plt.scatter(x1[:, 0], x1[:, 1], marker='+', color='red')

mean2 = [5, 8]
cov2 = [[2, 0], [0, 2]]
x2 = np.random.multivariate_normal(mean2, cov2, 200)
plt.scatter(x2[:, 0], x2[:, 1], marker='*', color='blue')

x_combined = np.concatenate([x1, x2])
theta = np.zeros((x_combined.shape[1], 1))
labels = np.array([0] * 200 + [1] * 200)

columns = ['X', 'Y']
df = pd.DataFrame(data=x_combined, columns=columns)

# df['lab'] = labels
#
# X = df.drop(['lab'], axis=1)
# y = df['lab']
test = df.sample(80)
train = df[~df.isin(test)]
train.dropna(inplace=True)

X_train, y_train, X_test, y_test = train.X, train.Y, test.X, test.Y

W = np.random.uniform(0, 1)
b = 0.1

parameters = fit(X_train, y_train, theta)
# x_values = [np.min(X[:, 1] - 5), np.max(X[:, 2] + 5)]
# y_values = - (parameters[0] + np.dot(parameters[1], x_values)) / parameters[2]

plt.plot(x1, x2, label='Decision Boundary')
plt.xlabel('Marks in 1st Exam')
plt.ylabel('Marks in 2nd Exam')
plt.legend()
plt.show()

print(accuracy(X_test, y_test.flatten()))
print("33")