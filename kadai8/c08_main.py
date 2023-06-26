#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import importlib
import c08_fun as fun
importlib.reload(fun)
import matplotlib.pylab as plt
import numpy as np
import scipy.fft
import scipy.signal

if __name__=="__main__":
    
    #%% 01
    fs = 256 # sampling frequency
    time_range = [0, 1] # sec
    t = np.arange(time_range[0], time_range[1], 1 / fs)
    params = [
        [.5, 25, .5 * np.pi], # amplitude, frequency, phase
        [.8, 5, -1 * np.pi],
        [1., 1, -.2 * np.pi],
        ]
    n_components = len(params)
    X = np.zeros([n_components, len(t)])
    for nn, [a, f, theta] in enumerate(params):
        X[nn] = fun.cos(t, a, f, theta)
    x = X.sum(0)
    plt.figure(1); plt.clf()
    plt.subplot(211)
    plt.plot(t, X.T)
    plt.subplot(212)
    plt.plot(t, x)
    plt.xlabel('Time [s]')
    plt.show()
    
    #%% 02
    f0 = 1.
    N = len(t)
    n_bases = 30
    coefs = fun.fourier_series(x, t, f0, N, n_bases) 
    x_inv = fun.fourier_series_reconstruct(t, f0, coefs)
    plt.figure(2); plt.clf()
    plt.plot(t, x, '-', label='observed')
    plt.plot(t, x_inv, '--', label='reconstructed')
    plt.legend()
    plt.xlabel('Time [s]')
    plt.show() 
    
    #%% 03
    X_f = scipy.fft.fft(x)
    freq_fft = scipy.fft.fftfreq(N, 1 / fs)
    freq_fs = fun.fsfreq(f0, n_bases)
    plt.figure(3); plt.clf()
    plt.subplot(211)
    plt.plot(freq_fs, coefs[:, 0] ** 2, label='a')
    plt.plot(freq_fs, coefs[:, 1] ** 2, label='b')
    plt.legend()
    plt.xlim([0, 30])
    plt.subplot(212)
    plt.plot(freq_fft[:N//2], (X_f * X_f.conj())[:N//2], label='X[f]')
    plt.xlim([0, 30])
    plt.xlabel('Frequency [Hz]')
    plt.legend()
    plt.show()
    
    #%% 04
    numtaps = 21
    cutoff = 10
    filter_coefs = scipy.signal.firwin(numtaps, cutoff, fs=fs)
    x_filtered = fun.convolve1d(filter_coefs, x)
    x_filtered_ref = scipy.signal.lfilter(filter_coefs, [1], x)
    plt.figure(4); plt.clf()
    plt.plot(t, x, label='observed')
    plt.plot(t[numtaps-1:], x_filtered, label='filtered')
    plt.plot(t, x_filtered_ref, '--', label='filtered (ref)')
    plt.legend()
    plt.xlabel('Time [s]')
    plt.show()
    
    #%% 05
    img = fun.read_img()
    plt.figure(5); plt.clf()
    plt.imshow(img)
    print('Image shape:\t{}'.format(img.shape))
    plt.show()
    
    #%% 06
    img = fun.to_gray(img)
    plt.figure(6); plt.clf()
    plt.imshow(img, cmap='gray')
    print('Image shape:\t{}'.format(img.shape))
    plt.show()
    
    #%% 07
    filter_size = [15, 15]
    filt = np.ones(filter_size) / np.prod(filter_size)
    img2 = fun.convolve2d(img, filt)
    img3 = scipy.signal.convolve2d(img, filt)
    plt.figure(7); plt.clf()
    plt.subplot(131)
    plt.imshow(img, cmap='gray')
    plt.subplot(132)
    plt.imshow(img2, cmap='gray')
    plt.subplot(133)
    plt.imshow(img3, cmap='gray')
    plt.show()