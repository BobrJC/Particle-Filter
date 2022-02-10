import enum
from settings import settings
from sympy import true
from main import main
import numpy as np
from spline import get_curves
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from copy import deepcopy


def check_errors(N, angle = False):
    settings_mult = deepcopy(settings)
    settings_mult['resample_type'] = 'mult'
    settings_syst = deepcopy(settings)
    settings_syst['resample_type'] = 'syst'
    settings_strat = deepcopy(settings)
    settings_strat['resample_type'] = 'strat'
    
    curves = get_curves()
    err_mult = np.zeros((len(curves) + 1, N, settings['iterations']))
    err_strat = np.zeros((len(curves) + 1, N, settings['iterations']))
    err_syst = np.zeros((len(curves) + 1, N, settings['iterations']))
    res_mult = np.zeros((len(curves) + 1,settings['iterations']))
    res_strat = np.zeros((len(curves) + 1,settings['iterations']))
    res_syst = np.zeros((len(curves) + 1,settings['iterations']))
    if angle:
        column = 1
    else:
        column = 0
    for i, curve in enumerate(curves):
        for test in range(N):
            err_syst[i][test] = main(settings_syst, curve)[:,column]
            err_mult[i][test] = main(settings_mult, curve)[:,column]
            err_strat[i][test] = main(settings_strat, curve)[:,column]
        res_mult[i] = np.sum(err_mult[i], axis = 0) / N
        res_strat[i] = np.sum(err_strat[i], axis = 0) / N
        res_syst[i] = np.sum(err_syst[i], axis = 0) / N
    #err_mult[len(curves)][test] = main(settings_mult)[:,column]
    # err_syst[len(curves)][test] = main(settings_syst)[:,column]
    # err_strat[len(curves)][test] = main(settings_strat)[:,column]
    # res_mult[len(curves)] = np.sum(err_mult[len(curves)], axis = 0) / N
    # res_strat[len(curves)] = np.sum(err_strat[len(curves)], axis = 0) / N
    # res_syst[len(curves)] = np.sum(err_syst[len(curves)], axis = 0) / N
    return res_mult, res_strat, res_syst


if __name__ == '__main__':
    #x = np.arange(1, 13)
    # fig, ax = plt.subplots(1, 2)
    start_time = time.perf_counter()
    # res = check_errors(10)
    # ax[0].plot(x, res[0][0], 'g', label='mult')
    # ax[0].plot(x, res[1][0],'b', label='strat')
    # ax[0].plot(x, res[2][0],'r', label='syst')
    # ax[1].plot(x, res[0][1], 'g', label='mult') 
    # ax[1].plot(x, res[1][1],'b', label='strat') 
    # ax[1].plot(x, res[2][1],'r', label='syst')
    # ax[0].legend()
    # ax[1].legend()
    #ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    result = []
    resamplings = ['_mult', '_strat', '_syst']
    curves = ['_curve', '_eight', '_model']
    # settings['noize_rot'] = 0.05
    # settings['noize_dist'] = 0.05
    # settings['noize_sens'] = 0.05
    preres = []
    index = 0
    path = r"//home//bobr_js//Particle filter//tests//100//"
    for N_p in range(25000, 25500, 500):
        settings['N_p'] = N_p
        preres.append(check_errors(100))
        
        for i in range(3):
            for k in range(3):
                with open(path + f'0.05_{N_p}' + f'{resamplings[i]}' + f'{curves[k]}', 'a') as f:
                    for j in range(settings['iterations']):
                        f.write(f'{preres[index][i][k][j]}\n')
        index += 1
        
    result.append(preres)
    
    # for noize in range(1, 3):
    #     settings['noize_rot'] = noize / 10
    #     settings['noize_dist'] = noize / 10
    #     settings['noize_sens'] = noize / 10
    #     preres = []
    #     index = 0
    #     for N_p in range(10000, 10500, 500):
    #         settings['N_p'] = N_p
    #         preres.append(check_errors(100))
    #         for i in range(3):
    #             for k in range(2):
    #                 with open(path + f'{noize/10}//' + f'{noize/10}_{N_p}' + f'{resamplings[i]}' + f'{curves[k]}', 'a') as f:
    #                     for j in range(settings['iterations']):
    #                         f.write(f'{preres[index][i][k][j]}\n')
    #         index+=1
    #     result.append(preres)

    for noize in range(2, 3):
        settings['noize_rot'] = noize / 10
        settings['noize_dist'] = noize / 10
        settings['noize_sens'] = noize / 10
        preres = []
        index = 0
        for N_p in range(15000, 15500, 500):
            settings['N_p'] = N_p
            preres.append(check_errors(100))
            for i in range(3):
                for k in range(2):
                    with open(path + f'{noize/10}//' + f'{noize/10}_{N_p}' + f'{resamplings[i]}' + f'{curves[k]}', 'a') as f:
                        for j in range(settings['iterations']):
                            f.write(f'{preres[index][i][k][j]}\n')
            index+=1
        result.append(preres)
    for noize in range(3, 5):
        settings['noize_rot'] = noize / 10
        settings['noize_dist'] = noize / 10
        settings['noize_sens'] = noize / 10
        preres = []
        index = 0
        for N_p in range(5000, 00, 500):
            settings['N_p'] = N_p
            preres.append(check_errors(100))
            for i in range(3):
                for k in range(2):
                    with open(path + f'{noize/10}//' + f'{noize/10}_{N_p}' + f'{resamplings[i]}' + f'{curves[k]}', 'a') as f:
                        for j in range(settings['iterations']):
                            f.write(f'{preres[index][i][k][j]}\n')
            index+=1
        result.append(preres)
    for noize in range(5, 6):
        settings['noize_rot'] = noize / 10
        settings['noize_dist'] = noize / 10
        settings['noize_sens'] = noize / 10
        preres = []
        index = 0
        for N_p in range(2000, 4500, 500):
            settings['N_p'] = N_p
            preres.append(check_errors(100))
            for i in range(3):
                for k in range(2):
                    with open(path + f'{noize/10}//' + f'{noize/10}_{N_p}' + f'{resamplings[i]}' + f'{curves[k]}', 'a') as f:
                        for j in range(settings['iterations']):
                            f.write(f'{preres[index][i][k][j]}\n')
            index+=1
        result.append(preres)
    print(time.perf_counter() - start_time)
    
    # fig.set_figwidth(20)
    # fig.set_figheight(15)
    # plt.show()
    