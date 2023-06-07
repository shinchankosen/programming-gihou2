import importlib
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
from statistics import mean, stdev
from scipy.stats import ttest_ind

def randomly_select(df, N):
    lis = df.sample(N)
    return (lis["Weight"].tolist(), lis["Height"].tolist())

def standarize(lis):
    mea = mean(lis)
    std = stdev(lis)
    return [(num - mea) / std for num in lis]

def estimate_y(h, a, b):
    return [a * num + b for num in h]

def compute_mse(w, est_w):
    return mean([pow(num - estnum, 2) for num, estnum in zip(w, est_w)])

def gradient_descent(h, w, alpha, max_iter, tolerance, a, b):
    mse = compute_mse(estimate_y(h, a, b), w)
    mse_list = [mse]
    for _ in range(max_iter):
        a -= alpha * mean(x * (a * x + b - y) for x, y in zip(h, w))
        b -= alpha * mean((a * x + b - y) for x, y in zip(h, w))
        new_mse = compute_mse(estimate_y(h, a, b), w)
        mse_list.append(new_mse)
        if abs(new_mse - mse) < tolerance:
            break
        mse = new_mse
    
    return (a, b, mse_list)

def gradient_descent_vector(h, w, alpha, max_iter, tolerance, a, b):
    # 行列の定義
    v = np.mat([a, b]).T
    X = []
    for x in h:
        X.append([x, 1])
    
    X = np.mat(X)
    y = np.mat([w]).T
    
    # mseの計算
    mse = compute_mse(estimate_y(h, a, b), w)
    mse_list = [mse]
    
    for _ in range(max_iter): 
        # mseを収束させるループ
        v -= alpha * np.dot(X.T, np.dot(X, v) - y) / len(h)
        new_mse = compute_mse(estimate_y(h, float(v[0][0]), float(v[1][0])), w)
        mse_list.append(new_mse)
        if abs(new_mse - mse) < tolerance:
            break
        mse = new_mse
    
    return (float(v[0][0]), float(v[1][0]), mse_list)

def least_squares(h, w):
    # 行列の定義
    X = []
    for x in h:
        X.append([x, 1])
    X = np.mat(X)
    y = np.mat([w]).T
    # 演算
    v = (X.T * X) ** (-1) * X.T * y
    return (float(v[0][0]), float(v[1][0]))

def out_ave_std_gender(df, lbl):
    values = df[lbl].tolist()
    gen = df["Gender"].tolist()
    Male = [val for g, val in zip(gen, values) if g == "Male"]
    Female = [val for g, val in zip(gen, values) if g == "Female"]
    p = ttest_ind(Male, Female, alternative='greater').pvalue
    return ((mean(Male), mean(Female)), (stdev(Male), stdev(Female)), p)