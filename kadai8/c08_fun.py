import importlib
import matplotlib.pylab as plt
import numpy as np
import scipy.fft
import scipy.signal
import cv2

def cos(t, a, f, theta):
    return a * np.cos(np.pi * 2 * f * t + theta)

def fourier_series(x, t, f0, N, n_bases):
    coefs = np.zeros([n_bases + 1, 2])
    for k in range(n_bases + 1):
        COS = [np.cos(np.pi * 2 * f0 * k * tn) for tn in t]
        SIN = [np.sin(np.pi * 2 * f0 * k * tn) for tn in t]
        coefs[k][0] = np.dot(COS, x) * (2 / N)
        coefs[k][1] = np.dot(SIN, x) * (2 / N)
        print("k =", k, *coefs[k])
    return coefs

def fourier_series_reconstruct(t, f0, coefs):
    n_bases = len(coefs) - 1
    x_inv = np.zeros(len(t))
    
    for n, tn in enumerate(t):
        COS = [np.cos(np.pi * 2 * f0 * (k + 1) * tn) for k in range(n_bases)]
        SIN = [np.sin(np.pi * 2 * f0 * (k + 1) * tn) for k in range(n_bases)]
        x_inv[n] = coefs[0][0] / 2 + np.dot(COS, coefs[1:, 0]) + np.dot(SIN, coefs[1:, 1])
    return x_inv

def fsfreq(f0, n_bases):
    return [f0 * i for i in range(n_bases + 1)] 

def convolve1d(flt, x):
    n = len(x) - len(flt) + 1
    return [sum(x[i + j] * val for i, val in enumerate(flt)) for j in range(n)]

def read_img():
    return np.asarray(cv2.cvtColor(cv2.imread('sophia.png'), cv2.COLOR_BGR2RGB))

def to_gray(img):
    return 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

def convolve2d(img, flt):
    n = len(img) - len(flt) - 1
    m = len(img[0]) - len(flt[0]) - 1
    return [[sum(val * img[i + y][j + x] for y, vec in enumerate(flt) 
                 for x, val in enumerate(vec)) for j in range(m)] for i in range(n)]
