import numpy as np
import matplotlib.pyplot as plt


def random_split(data, test_points):
    """return two matrices splitting the data at random3"""
    ranks = np.arange(data.shape[0])
    np.random.shuffle(ranks)
    train = data[ranks >= test_points, :]
    test = data[ranks < test_points, :]
    return train, test

def polyFit(degree,x,y,color):
    coefs = np.polyfit(x, y, degree)
    pxs = np.linspace(0, max(x), 100)
    poly = np.polyval(coefs, pxs)
    plt.plot(pxs, poly, color)





np.set_printoptions(precision=2)
data = np.loadtxt('bluegills.txt', delimiter='\t', skiprows=1)
Ys = data[:, 1]
Xs = data[:, 0]

means = np.mean(Xs)
stdevs = np.std(Xs)
Xs = (Xs - means) / stdevs
means = np.mean(Ys)
stdevs = np.std(Ys)
Ys = (Ys - means) / stdevs


data = np.array((Xs, Ys)).T
#scale = np.max(data,axis=0)
#data = data/scale  scale then expand check page 28 of lecture notes
train, temp = random_split(data, 39)
valid, test = random_split(temp, 19)

x, y = (train[:, 0], train[:, 1])
plt.plot(x, y, '+r')
plt.plot(valid[:, 0],valid[:, 0],'sg')
plt.plot(test[:, 0],test[:, 0],'^b')
plt.figure(1, figsize=(12, 8), frameon=False)
plt.axis([-3, 3, -4, 2])
plt.title("Blue gill size")


colors = ['-b','-r','-g','-c','-y','-m']
for degree in range(1, 7):
    polyFit(degree, x, y, colors[degree-1])
plt.show()






#train, temp = random_split(data, )

