from  sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import random
from scipy.spatial.distance import pdist, squareform

s = np.array([[i/10] for i in range(250)])
initial = random.sample(range(1, 250), 5)
sample = s[initial]
# matrix = squareform(pdist(sample))
# value = np.min(matrix[np.nonzero(matrix)])
value = np.min(pdist(sample))
end = sample
print(value)
for i in range(10000):
    select = random.sample(range(1, 250), 5)
    sample = s[select]
    matrix = squareform(pdist(sample))
    if np.min(pdist(sample)) > value:
        value = np.min(pdist(sample))
        print(value)
        end = sample
        
def function_test(x):
    y = x*np.cos(x) + np.exp(-x)
    return  np.around(y, decimals=4)
def EI(x):
    f_min = np.min(y)
    pred = np.around(np.array([i[0] for i in clf.predict(x)]), decimals=4)
    sig = clf.predict(x, return_std = True)[1]
#     print(sig)
    args0 = (f_min - pred) / sig
    args1 = (f_min - pred) * norm.cdf(args0)

    args2 = sig * norm.pdf(args0)

    ei = args1 + args2
    ei = np.nan_to_num(ei)
    return ei
points = np.array([[i/10] for i in range(250)])
true_y = function_test(np.array([[i/10] for i in range(250)]))

x = end
y = function_test(x)


clf = GaussianProcessRegressor()
clf.fit(x, y)
index_best = np.argmax(EI(points))
index_best = (-EI(points)).argsort()[:1]
# print(index_best)

n_iter = 10

fig = plt.figure(figsize=[10, 10])

for i in range(n_iter):
    ax1 = fig.add_subplot((n_iter + 1) // 2, 2, i + 1)
    ax2 = ax1.twinx()  # this is the important function

    data,  = ax1.plot(x, y,linestyle='', marker='o', color='orange')
    (predicted, ) = ax1.plot(points, clf.predict([[i/10] for i in range(250)]), color = 'red')
    (true_func, ) = ax1.plot(points, true_y, color = 'blue')
    (ei, ) = ax2.plot(points, EI(points), color = 'green')
    (opt,) = ax1.plot(
            points[index_best], true_y[index_best], linestyle="", marker="*", color="r"
        )
    lines = [true_func, data, predicted,  opt, ei]
    fig.suptitle("EGO optimization of $f(x) = x \cos{x} + \exp{x}$")
    fig.subplots_adjust(hspace=0.4, wspace=0.4, top=0.8)
    ax1.set_title("iteration {}".format(i + 1))
    fig.legend(
        lines,
        [
            "f(x)=xcos(x) + exp(x)",
            "Given data points",
            "Prediction",
#             "Kriging 99% confidence interval",
            "Next point to evaluate",
            "Expected improvment function",
        ],
    )

    x = np.append(x ,points[index_best], axis=0)
    y = function_test(x)
    clf = GaussianProcessRegressor()
    clf.fit(x, y)
    index_best = np.argmax(EI(points))

    index_best = (-EI(points)).argsort()[:1]
#     print(index_best)

