import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from scipy import stats
def problem_1a (A, B):
    return A + B

def problem_1b (A, B, C):
    return np.dot(A,B) - C

def problem_1c (A, B, C):
    return A*B + np.transpose(C)

def problem_1d (x, y):
    return np.dot(np.transpose(x),y)

def problem_1e (A, x):
    return np.linalg.solve(A,x)

def problem_1f (A, x):
    return np.transpose(np.linalg.solve(np.transpose(A),(x)))

def problem_1g (A, i):
    return np.sum(A[i,0::2])

def problem_1h (A, c, d):
    return np.mean(A[np.nonzero((c<=A)&(A<=d))])

def problem_1i (A, k):
    w,v = np.linalg.eig(A)
    col = w.argsort()[-k:][::-1]
    return v[:,col]

def problem_1j (x, k, m, s):
    z = np.ones((np.size(x),1),dtype= int)
    i = np.identity((np.size(x)),dtype = int)
    temp = np.empty((np.size(x),k))

    for j in range(k): 
        temp[:,j] = np.array(np.random.multivariate_normal((x + m*z).flatten(),s*i))
    return temp

def problem_1k (A):
    rng = np.random.default_rng()
    return rng.permutation(A,axis = 0)

def problem_1l (x):
    return (x - np.mean(x))/np.std(x)

def problem_1m (x, k):
    return np.reshape(np.repeat(x, k), (np.size(x),k))

def problem_1n (X):
    M = np.atleast_3d(X)
    M = np.repeat(M, M.shape[1], axis=-1)
    N = np.transpose(M, (0, 2, 1))
    return np.linalg.norm(M-N, axis=0)

def linear_regression (X_tr, y_tr):
    return np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X_tr),X_tr)),np.transpose(X_tr)),y_tr)

def train_age_regressor ():
    # Load data
    X_tr = np.reshape(np.load("age_regression_Xtr.npy"), (-1, 48*48))
    ytr = np.load("age_regression_ytr.npy")
    X_te = np.reshape(np.load("age_regression_Xte.npy"), (-1, 48*48))
    yte = np.load("age_regression_yte.npy")

    w = linear_regression(X_tr, ytr)
    y_pred = np.dot(X_tr,w)
    fmse_train = (np.sum(np.square((y_pred - ytr))))/(2*len(ytr))
    y_pred = np.dot(X_te,w)
    fmse_test = (np.sum(np.square((y_pred - yte))))/(2*len(yte))
    print('Train:',fmse_train,'\nTest:', fmse_test)
    # Report fMSE cost on the training and testing data (separately)
    # ...

A = np.array([[1, 2, 5], 
        [-5, 8, 9],
        [3, 2, 4]])


B = np.array([[3, 1, 2], 
            [9, -3, 8],
            [2, 0, 1]])

C = np.array([[2, 1, -2], 
            [6, -1, 9],
            [4, 1, 3]])
x = np.array([[1],[2],[3]])
y = np.array([[4],[8],[15]])

def problem_3a():
  
    poisson_data = np.load("PoissonX.npy")
    # plot the histogram
    plt.hist(poisson_data, density=True,label='Data')
    plt.legend()
    plt.show()

    # using spcipy
    mu = [2.5, 3.1, 3.7, 4.3]
    count = 1
    fig, ax = plt.subplots(2, 2, figsize=(
        24, 32))
    for m in mu:
        data_poisson = poisson.rvs(mu=m, size=10000)
        plt.subplot(2, 2, count)
        ax = plt.hist(data_poisson, density=True,label = m)
        count = count + 1
        plt.legend()


    plt.show()

def problem_3b():

    arr = [-2,-1,-0.2,0.2,0.4, 0.7,1,1.5 ]
    for x in arr:
        mu = x*x
        variance = math.pow(2 - 1/(1+np.exp(x*x*-1)), 2)
        sigma = math.sqrt(variance)
        e = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
        plt.plot(e, stats.norm.pdf(e, mu, sigma),label=x)
        plt.legend()
    plt.show()

print(problem_1a(A,B))
print(problem_1b(A,B,C))
print(problem_1c(A,B,C))
print(problem_1d(x,y))
print(problem_1e(A,x))
print(problem_1f(A,x))
print(problem_1g(A,1))
print(problem_1h(A,1,9))
print(problem_1i(A,1))
print(problem_1j(x,3,3,2))
print(problem_1k(A))
print(problem_1l(y))
print(problem_1m(y,6))
print(problem_1n(A))
train_age_regressor ()
problem_3a()
problem_3b()





