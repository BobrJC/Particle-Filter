import enum
from importlib.util import spec_from_file_location
from settings import settings, resamplings, curves
from sympy import true
from main import main
import numpy as np
from spline import get_curves
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from copy import copy, deepcopy
from tools import get_script_dir, update_folder
from Robot import model
import os


def check_errors(N, angle = False, R_model = None):
    settings_mult = deepcopy(settings)
    settings_mult['resample_type'] = 'mult'
    settings_syst = deepcopy(settings)
    settings_syst['resample_type'] = 'syst'
    settings_strat = deepcopy(settings)
    settings_strat['resample_type'] = 'strat'
    iterations = settings['iterations']
    if settings['detail']:
        iterations*=2
    if R_model is None:
        curves = get_curves()
    else:
        curves = [None]
    iterations+=1
    
    err_mult = np.zeros((len(curves), N, iterations))
    err_strat = np.zeros((len(curves), N, iterations))
    err_syst = np.zeros((len(curves), N, iterations))
    res_mult = np.zeros((len(curves), iterations))
    res_strat = np.zeros((len(curves), iterations))
    res_syst = np.zeros((len(curves), iterations))
    if angle:
        column = 1
    else:
        column = 0
    for i, curve in enumerate(curves):
        for test in range(N):
            err_syst[i][test] = np.asarray(main(settings_syst, curve, R_model))[:,column]
            err_mult[i][test] = np.asarray(main(settings_mult, curve, R_model))[:,column]
            err_strat[i][test] = np.asarray(main(settings_strat, curve, R_model))[:,column]
        res_mult[i] = np.sum(err_mult[i], axis = 0) / N
        res_strat[i] = np.sum(err_strat[i], axis = 0) / N
        res_syst[i] = np.sum(err_syst[i], axis = 0) / N
    return res_mult, res_strat, res_syst

def test_curves(N_p_start, N_p_end, N_p_step, N, noize_start, noize_end, noize_step, model = None):
    result = []
    preres = []
    index = 0
    script_dir = get_script_dir() 
    update_folder('/tests')
    update_folder(f'/tests/{N}')
    path = rf'{script_dir}/tests/{N}/'
    if model is None:
        trajectories = curves
    else:
        trajectories = [model.get_name()]
    for i in resamplings:
        update_folder(f'/tests/{N}/' + i)
        for j in np.arange(noize_start, noize_end, noize_step):
            update_folder(f'/tests/{N}/' + i + '/' + j.__str__())
            for k in trajectories:
                update_folder(f'/tests/{N}/' + i + '/' + j.__str__() + '/' + k)
    for noize in np.arange(noize_start, noize_end, noize_step):
        settings['noize_rot'] = noize
        settings['noize_dist'] = noize
        settings['noize_sens'] = noize
        preres = []
        index = 0
        for N_p in range(N_p_start, N_p_end, N_p_step):
            settings['N_p'] = N_p
            start_time = time.perf_counter()
            preres.append(check_errors(N, R_model=model))
            print(preres)
            print('not model time:', time.perf_counter() - start_time)
            for i in range(len(resamplings)):
                for k in range(len(trajectories)):
                    cur_path = path + f'{resamplings[i]}/' + f'{noize}/' + f'{trajectories[k]}/' + f'{N_p}'
                    if os.path.exists(cur_path):
                        os.remove(cur_path)
                    with open(cur_path, 'a') as f:
                        for j in range(settings['iterations']):
                            f.write(f'{preres[index][i][k][j + 1]}\n')
            index+=1
        result.append(preres)
    return result

def test_models_new(N_p_start, N_p_end, N_p_step, N, noize_start, noize_end, noize_step, model):
    result = []
    script_dir = get_script_dir() 
    path = rf"{script_dir}//tests//{N}//models//"
    noize_n = 0
    for noize in np.arange(noize_start, noize_end, noize_step):
        settings['noize_rot'] = noize
        settings['noize_dist'] = noize
        settings['noize_sens'] = noize
        preres = []
        N_p_n = 0
        for N_p in range(N_p_start, N_p_end, N_p_step):
            settings['N_p'] = N_p
            preres.append([])
            re = check_errors(N, R_model=model)
            preres[N_p_n].append(re)
            for i in range(len(resamplings)):
                with open(path + f'{resamplings}//' + f'{N_p}' + f'_{resamplings[i]}' + f'_speed{round(model.get_speed(), 1)}', 'a') as f:
                    for j in range(settings['iterations']):
                        f.write(f'{preres[noize_n][N_p_n][i][0][j]}\n')
            N_p_n += 1
        noize_n += 1
        result.append(preres)

# def test_models(N_p_start, N_p_end, N_p_step, N, noize_start, noize_end, noize_step, speed_start, speed_end, speed_step):
#     result = []
#     preres = []
#     index = 0
#     script_dir = get_script_dir() 
#     path = rf"{script_dir}//tests//{N}//models//"
#     for noize in np.arange(noize_start, noize_end, noize_step):
#         settings['noize_rot'] = noize
#         settings['noize_dist'] = noize
#         settings['noize_sens'] = noize
#         preres = []
#         index = 0
#         for N_p in range(N_p_start, N_p_end, N_p_step):
#             preres.append([])
#             settings['N_p'] = N_p
#             for speed in np.arange(speed_start, speed_end, speed_step):
#                 j = 0
#                 start_time = time.perf_counter()
#                 preres[index].append(check_errors(N, R_model=model(.4, .4, 180, speed)))
#                 print('model time:', time.perf_counter() - start_time, 'N_p=', N_p, )
#                 print(preres[index][j])
#                 for i in range(len(resamplings)):
#                     with open(path + f'{round(noize, 2)}//' + f'{N_p}' + f'_{resamplings[i]}' + f'_speed{round(speed, 1)}', 'a') as f:
#                         for k in range(settings['iterations']):
#                             f.write(f'{preres[index][j][i][k]}\n')
#                     j+=1
#             index+=1
#         result.append(preres)

def visualize_errors(N, noize, N_p, model = False, speeds = None, save = False, pathh = None):
    script_dir = get_script_dir() 
    errors = []
    path = rf"{script_dir}//tests//{N}//"

    if type(N_p) is not list:
        if pathh is None:
            if not model:
                for i, curve in enumerate(curves):
                    errors.append([])
                    for j, resampling in enumerate(resamplings):
                        with open(path + f'{noize}//' + f'{noize}' + f'_{N_p}' + f'_{resampling}' + f'_{curve}', 'r') as f:
                            err_list = [float(err.strip()) for err in f]
                        errors[i].append(err_list)
            else:
            
                for i, resampling in enumerate(resamplings):
                        errors.append([])
                        if speeds is not None:
                            for speed in speeds:
                                with open(path + 'models//' + f'{noize}//' + f'{N_p}' + f'_{resampling}' + f'_speed{speed}') as f:
                                    err_list = [float(err.strip()) for err in f]
                                errors[i].append(err_list)
        else:
            print('here')
            
            for i, p in enumerate(pathh):
                errors.append([])
                print(p)
                with open(path + p) as f:
                    errors[i] = [float(err.strip()) for err in f]
    else:
        if not model:
            for i, Np in enumerate(N_p):
                errors.append([])
                #for j, resampling in enumerate(resamplings):
                with open(path + f'{noize}//' + f'{noize}' + f'_{Np}' + '_mult' + '_eight') as f:
                    err_list = [float(err.strip()) for err in f]
                errors[i] = err_list
        else:
            for i, Np in enumerate(N_p):
                errors.append([])
                #for j, resampling in enumerate(resamplings):
                with open(path + 'models//' + f'{noize}//' + f'{Np}' + '_strat' + '_speed0.5') as f:
                    err_lis = [float(err.strip()) for err in f]
                    err_list = []
                    for j, err in enumerate(err_lis):
                        err_list.append(err)
                        if j == 11:
                            break
                errors[i] = err_list

        
    x = np.arange(1, 13)
    if type(N_p) is not list:
        if not model and path is None:
            fig, ax = plt.subplots(1, len(curves))
            for i in range(len(curves)):
                for j in range(len(resamplings)):
                    ax[i].plot(x, errors[i][j], label=resamplings[j])
                
                ax[i].legend(prop={'size': 30})
        elif path is None:
            fig, ax = plt.subplots(1, len(speeds))
            for i in range(len(resamplings)):
                if len(speeds) > 1:
                    for j in range(len(speeds)):
                        ax[i].plot(x, errors[i][j], label=resamplings[j])
                    ax[i].legend()
                else:
                    ax.plot(errors[i][0])
                    ax.legend(prop={'size': 30})
        else:
            fig, ax = plt.subplots()
            for i in range(len(pathh)):
                ax.plot(x, errors[i], label=noize[i])
            ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
            ax.legend(prop={'size': 30})
    else:
        fig, ax = plt.subplots()
        print(errors)
        for i in range(len(N_p)):
            ax.plot(x, errors[i], label=f'ПРЧ = {int(N_p[i]/100)}')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.legend(prop={'size': 20})

    fig.set_figwidth(20)
    fig.set_figheight(15)
    plt.show()

if __name__ == '__main__':
    settings['visualize'] = False

    # test_models(500, 3500, 500, 70, .05, .2, .05, .5, 1, .5)
    # start_time = time.perf_counter()
    # test_models(1500, 5000, 500, 70, .05, .2, .05, 1, 2, .5)
    # print(time.perf_counter() - start_time)
    # print(time.perf_counter() - start_time)
    # test_models(2000, 5000, 500, 70, .2, .3, .1, .5, 2, .5)
    # print(time.perf_counter() - start_time)
    # test_models(4500, 7000, 500, 50, .2, .3, .1, 2, 3.5, .5)
    # print(time.perf_counter() - start_time)
    # test_models(500, 2500, 500, 70, .3, .6, .1, .5, 3.5, .5)
    # test_models(5000, 5500, 500, 70, .05, .1, .05, .5, 1.5, .5)
    # #visualize_errors(1, 0.1, 500)
    # test_models(8500, 10000, 500, 40, .05, .15, .05, 2, 3.5, .5)
    # test_models(5000, 9000, 500, 40, .05, .15, .05, 2, 3.5, .5)
    # visualize_errors(100, ['strat 0.1','syst 0.1', 'mult 0.1', 'strat 0.2','syst 0.2', 'mult 0.2'], 3500, 
    #                 pathh=['0.1//0.1_5000_strat_curve',
    #                         '0.1//0.1_5000_syst_curve', 
    #                         '0.1//0.1_5000_mult_curve',
    #                         '0.2//0.2_5000_strat_curve',
    #                         '0.2//0.2_5000_syst_curve', 
    #                         '0.2//0.2_5000_mult_curve'])
    # visualize_errors(100, 0.1, [i for i in range(3500, 10500, 500)])
    #N_p = [i for i in range(500, 3500, 500)]
    #visualize_errors(100, 0.1, N_p, model=True)
    
    # start_time = time.perf_counter()
    test_curves(2000, 2100, 100, 1, 0.1, 0.2, 0.1, model=model(.4, .4, 180, 2))
    #print(check_errors(1))
    # print(time.perf_counter() - start_time)
    # start_time = time.perf_counter()
    #test_models_new(2000, 2100, 100, 1, 0.5, 0.6, 0.1, model(.4, .4, 180, 1))
    # test_models_new(2000, 2100, 100, 1, 0.5, 0.6, 0.1, model(.4, .4, 180, .5))
    # print(time.perf_counter() - start_time)
    #print(time.perf_counter() - start_time)
    
    