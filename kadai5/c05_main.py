import importlib
import c05_fun as fun
importlib.reload(fun)
import pandas as pd
import matplotlib.pylab as plt
import numpy as np

if __name__=="__main__":
    
    #%%
    # Download the csv file
    # If you have error with these codes, directly download the csv file from the URL.
    import urllib.request
    url = 'https://gist.githubusercontent.com/nstokoe/7d4717e96c21b8ad04ec91f361b000cb/raw/bf95a2e30fceb9f2ae990eac8379fc7d844a0196/weight-height.csv'
    save_file = 'weight-height.csv'
    urllib.request.urlretrieve(url, save_file)
    df = pd.read_csv(save_file)
    
    #%% 05_01
    # Plot samples
    N = 100
    w, h = fun.randomly_select(df, N)
    plt.figure(1).clf()
    plt.scatter(w, h)
    plt.xlabel('Height')
    plt.ylabel('Weight')
    
    #%% 05_02
    # Standarized
    w = fun.standarize(w)
    h = fun.standarize(h)
    plt.figure(2).clf()
    plt.scatter(w, h, label='Observed')
    print('Averaged standardized w:\t{:.2f}+-{:.2f}'.format(np.mean(w), np.std(w))) # np.mean(w) = 0, np.std(w) = 1
    print('Averaged standardized h:\t{:.2f}+-{:.2f}'.format(np.mean(h), np.std(h))) # np.mean(h) = 0, np.std(h) = 1
    
    #%% 05_03
    # MSE
    a = 2
    b = 1
    est_w = fun.estimate_y(h, a, b)
    plt.figure(2)
    plt.scatter(est_w, h, label='Estimated 03')
    mse = fun.compute_mse(w, est_w)
    print('03:\tMSE: {:.2f}\ta: {:.2f}\tb: {:.2f}'.format(mse, a, b))
    
    
    #%% 05_04
    # Gradient descent
    alpha = .01 # learning rate
    max_iter = 100 # maximum number of iterations
    tolerance = 1e-6 # minimum MSE difference between two iterations
    init_a = 0. # initial value for a
    init_b = 0. # initial value for b
    a_gd, b_gd, ep_mse = fun.gradient_descent(h, w, alpha, max_iter, tolerance, init_a, init_b)
    est_w = fun.estimate_y(h, a_gd, b_gd)
    plt.scatter(est_w, h, label='Estimated 04')
    mse = fun.compute_mse(w, est_w)
    print('04:\tMSE: {:.2f}\ta: {:.2f}\tb: {:.2f}'.format(mse, a_gd, b_gd))
    plt.figure(3).clf()
    plt.plot(ep_mse, label='04')
    
    
    
    #%% 05_05
    # Gradient descent in vector form
    a_gdv, b_gdv, ep_mse = fun.gradient_descent_vector(h, w, alpha, max_iter, tolerance, init_a, init_b)
    est_w = fun.estimate_y(h, a_gdv, b_gdv)
    plt.figure(2)
    plt.scatter(est_w, h, label='Estimated 05')
    mse = fun.compute_mse(w, est_w)
    print('05:\tMSE: {:.2f}\ta: {:.2f}\tb: {:.2f}'.format(mse, a_gdv, b_gdv))
    plt.figure(3)
    plt.plot(ep_mse, label='05')
    plt.xlabel('Iterations')
    plt.ylabel('MSE')
    plt.legend()
    
    
    #%% 05_06
    # Least squares
    a_ls, b_ls = fun.least_squares(h, w)
    est_w = fun.estimate_y(h, a_ls, b_ls)
    plt.figure(2)
    plt.scatter(est_w, h, label='Estimated 06')
    mse = fun.compute_mse(w, est_w)
    print('06:\tMSE: {:.2f}\ta: {:.2f}\tb: {:.2f}'.format(mse, a_ls, b_ls))
    
    #%% 05_07
    # numpy.polyfit
    a_pf, b_pf = np.polyfit(h, w, 1)
    est_w = fun.estimate_y(h, a_pf, b_pf)
    plt.figure(2)
    plt.scatter(est_w, h, label='Estimated 07')
    mse = fun.compute_mse(w, est_w)
    print('07:\tMSE: {:.2f}\ta: {:.2f}\tb: {:.2f}'.format(mse, a_pf, b_pf))
  
    plt.xlabel('Standardized height')
    plt.ylabel('Standardized weight')
    plt.legend()
    
    #%% 05_08
    ave_h, std_h, pvalue_h = fun.out_ave_std_gender(df, 'Height')
    ave_w, std_w, pvalue_w = fun.out_ave_std_gender(df, 'Weight')
    plt.figure(4).clf()
    plt.subplot(121)
    plt.bar([0, 1], ave_h, yerr=std_h)
    plt.ylabel('Height')
    plt.xticks([0, 1], ['Male', 'Female'])
    
    plt.subplot(122)
    plt.bar([0, 1], ave_w, yerr=std_w)
    plt.ylabel('Weight')
    plt.xticks([0, 1], ['Male', 'Female'])
    plt.tight_layout()
    
    if pvalue_h > .95:
        print('Female is higher than Male (p={:.3f})'.format(1 - pvalue_h))
    elif pvalue_h < .05:
        print('Male is higher than Female (p={:.3f})'.format(pvalue_h))
    else:
        print('There is no significant difference between Male and Female in height (p={:.3f})'.format(pvalue_h))
        
    if pvalue_w > .95:
        print('Female is heavier than Male (p={:.3f})'.format(1 - pvalue_w))
    elif pvalue_w < .05:
        print('Male is heavier than Female (p={:.3f})'.format(pvalue_w))
    else:
        print('There is no significant difference between Male and Female in weight (p={:.3f})'.format(pvalue_w))
    
    plt.show()