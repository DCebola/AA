import numpy as np
import matplotlib.pyplot as plt


def random_split(_data, test_points):
    """return two matrices splitting the data at random3"""
    ranks = np.arange(_data.shape[0])
    np.random.shuffle(ranks)
    _train = _data[ranks >= test_points, :]
    _test = _data[ranks < test_points, :]
    return _train, _test


def polyFit(_degree, _x, _y, color, _pxs):
    _coefs = np.polyfit(_x, _y, _degree)
    poly = np.polyval(_coefs, _pxs)
    plt.plot(_pxs, poly, color)
    return _coefs


def mean_square_error(_data, _coefs):
    pred = np.polyval(_coefs, _data[:, 0])
    error = np.mean((_data[:, 1] - pred) ** 2)
    return error


np.set_printoptions(precision=2)
data = np.loadtxt('yield.txt', delimiter='\t', skiprows=1)

Ys = data[:, 1]
Xs = data[:, 0]

x_means = np.mean(Xs)
x_stdevs = np.std(Xs)
Xs = (Xs - x_means) / x_stdevs

y_means = np.mean(Ys)
y_stdevs = np.std(Ys)
Ys = (Ys - y_means) / y_stdevs

data = np.array((Xs, Ys)).T

data_size = data.shape[0]
train, temp = random_split(data, data_size / 2)
valid, test = random_split(temp, data_size / 4)

pxs = np.linspace(min(data[:, 0]), max(data[:, 0]), 100)
x, y = (train[:, 0], train[:, 1])
plt.plot(x, y, '.r')
plt.plot(valid[:, 0], valid[:, 0], 'sg')
plt.plot(test[:, 0], test[:, 0], '^b')
plt.figure(1, figsize=(12, 8), frameon=False)
plt.axis()
plt.title("Blue gill size")

colors = ['-b', '-r', '-g', '-c', '-y', '-m']
best_error = 10000000
best_coefs = []
best_degree = -1
for degree in range(1, 7):
    coefs = polyFit(degree, x, y, colors[degree - 1], pxs)
    valid_error = mean_square_error(valid, coefs)
    if valid_error < best_error:
        best_error = valid_error
        best_coefs = coefs
        best_degree = degree

test_error = mean_square_error(test, best_coefs)
print("Degree: " + str(best_degree))
print("Test error: " + str(test_error))
plt.show()

# train, temp = random_split(data, )
