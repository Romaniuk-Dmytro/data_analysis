import numpy as np
import matplotlib.pyplot as plt
import random


def get_derivative(train_value, error):
    return -2 * (train_value * error), -2 * error


noise = np.random.normal(0, 1, 200)
x = np.random.uniform(0, 3, 200)

a = 2
b = -4

Y = a * x + b + noise

plt.scatter(x, Y, marker='.', color='red')
plt.plot()
plt.plot(x, a * x + b)
# plt.show()

array = np.arange(200)
np.random.shuffle(array)

x_train = x[array]
y_train = Y[array]

step = 0.01
for _ in range(200):
    test_y = a * x_train + b
    e = y_train - test_y

    a_deriv, b_deriv = get_derivative(x_train, e)

    a = a - step * a_deriv
    b = b - step * b_deriv

print(a, b)
