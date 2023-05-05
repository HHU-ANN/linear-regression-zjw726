# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def ridge(data):
    x, y = read_data()
    ep=np.eye(6)
    a=0.5
    w = np.dot(np.linalg.inv(np.dot(x.T, x)+a*ep), np.dot(x.T, y))
    return data @ w
    
def lasso(data):
    x, y = read_data()
    a = 0.01
    ep = 10000
    lr = 0.001
    w = np.zeros(x.shape[1])
    for i in range(ep):
        grad = np.matmul(x.T, np.matmul(x, w) - y) / x.shape[0] + a * np.sign(w)
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1:
            grad /= grad_norm
        w -= lr * grad
    return data @ w
                 
def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y
