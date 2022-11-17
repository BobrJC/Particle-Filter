from Particle_filter import ParticleFilter
import settings
from main import main
import numpy as np
from spline import get_curves
import time
from copy import deepcopy
from tools import get_script_dir, update_folder
from Robot import model
import os

# Функция проведения тестов с текущими настройками.
# Принимает количество повторений теста. Тип данных int. 
# Опционально принимает модель перемещения робота. Тип - класс model.
# Возвращает два списка - с усредененными ошибками на каждой итерации и с временем выполнения.
def check_errors(N, R_model = None, settings = settings.settings, 
                test_resamplings: dict = settings.test_resamplings, 
                change_n = settings.change_n, curves = [None]):
    iterations = settings['iterations']
    test_settings = []
    err_list = []
    res_list = []
    time_arr = []
    if settings['detail']:
        iterations*=2
    if R_model is not None:
        curves = [None]
    iterations+=1
    for type in change_n:
        for resampling in test_resamplings.values():
            settings['change_n'] = type
            settings['resample_type'] = resampling
            test_settings.append(deepcopy(settings))
            if settings['change_n'] == None:
                err_list.append(np.zeros((len(curves), N, iterations)))
                res_list.append(np.zeros((len(curves), iterations)))
                time_arr.append(np.zeros((len(curves))))
            else:
                err_list.append(np.zeros((len(curves), N, 2, iterations)))
                res_list.append(np.zeros((len(curves), 2, iterations)))
                time_arr.append(np.zeros((len(curves))))
    for i, curve in enumerate(curves):
        for test in range(N):
            for j, setting in enumerate(test_settings):
                time_start = time.perf_counter()
                try:
                    err_list[j][i][test] = np.asarray(main(setting, curve, R_model))
                except FloatingPointError:
                    err_list[j][i][test].fill(1)
                time_end = time.perf_counter()
                time_arr[j][i] += time_end - time_start
        for j, res in enumerate(res_list):
            res_list[j][i] = np.sum(err_list[j][i], axis = 0) / N
            time_arr[j][i] /= N
    return res_list, time_arr

# Функция задающая параметры тестирования и сохраняющая результаты. 
# Принимает число частиц для начала диапозона тестирования, для конца диапозона, 
# шаг изменения числа частиц, число тестирований, шум налача диапозона тестирования, 
# конца диапозона, шаг изменения шума. тип данных float. 
# Опционально принимает модель перемещения робота. Тип - класс model.
# Возвращает список со всеми результатами тестирований.
def test_curves(N_p_start, N_p_end, N_p_step, N, 
                noize_start, noize_end, noize_step, 
                model = None, settings = settings.settings, 
                test_resamplings = settings.test_resamplings, 
                change_n = settings.change_n, curves = settings.curves):

    result = []
    trajectories = []
    index = 0
    script_dir = get_script_dir() 
    update_folder('/tests')
    update_folder(f'/tests/{N}')
    path = rf'{script_dir}/tests/{N}/'
    if model is None:
        default_curv = get_curves()
        for tr in curves:
            if (tr == default_curv[0]).any():
                trajectories.append("curve")
            elif (tr == default_curv[1]).any():
                trajectories.append("eight")
            else:
                trajectories.insert(0, "generated")
    else:
        trajectories = [model.get_name()]
    if settings['change_n'] != None:
        resampling_names = []
        for i in test_resamplings.keys():
            for j in change_n:
                resampling_names.append(i + j)
    else:
        resampling_names = list(test_resamplings.keys())
    for i in resampling_names:
        update_folder(f'/tests/{N}/' + 'New' + i)
        for j in np.arange(noize_start, noize_end, noize_step):
            update_folder(f'/tests/{N}/' + 'New' + i + '/' + str(j))
            for k in trajectories:
                update_folder(f'/tests/{N}/' + 'New' + i + '/' + str(j) + '/' + k)
    for noize in np.arange(noize_start, noize_end, noize_step):
        settings['noize_rot'] = noize
        settings['noize_dist'] = noize
        settings['noize_sens'] = noize
        preres = []
        index = 0
        for N_p in range(N_p_start, N_p_end, N_p_step):
            settings['N_p'] = N_p
            start_time = time.perf_counter()
            print('sett', settings, 'res ', test_resamplings, 'ch', change_n)
            preres.append(check_errors(N, R_model=model, settings=settings, test_resamplings=test_resamplings, change_n=change_n, curves=curves))
            end_time = time.perf_counter() - start_time
            for i in range(len(resampling_names)):
                for k in range(len(trajectories)):
                    cur_path = path + f'New{resampling_names[i]}/' + f'{noize}/' + f'{trajectories[k]}/' + f'{N_p}'
                    if os.path.exists(cur_path):
                        os.remove(cur_path)
                    if settings['change_n'] == None:
                        with open(cur_path, 'a') as f:
                            for j in range(settings['iterations']):
                                f.write(f'{preres[index][0][i][k][j + 1]}\n')
                    else:
                        cur_path = path + f'New{resampling_names[i]}/' + f'{noize}/' + f'{trajectories[k]}/' + f'{N_p}'
                        if os.path.exists(cur_path):
                            os.remove(cur_path)
                        with open(cur_path, 'a') as f:
                            for j in range(settings['iterations']):
                                f.write(f'{preres[index][0][i][k][0][j + 1]}\n')
                        cur_path = path + f'New{resampling_names[i]}/' + f'{noize}/' + f'{trajectories[k]}/' + f'{N_p}' + '_N'
                        if os.path.exists(cur_path):
                            os.remove(cur_path)
                        with open(cur_path, 'a') as f:
                            for j in range(settings['iterations']):
                                f.write(f'{preres[index][0][i][k][1][j + 1]}\n')
                    cur_path = path + f'New{resampling_names[i]}/' + f'{noize}/' + f'{trajectories[k]}/' + f'{N_p}' + '_time'
                    if os.path.exists(cur_path):
                        os.remove(cur_path)
                    with open(cur_path, 'a') as f:
                            f.write(str(preres[index][1][i][k]))
            index+=1
        result.append(preres)
    return result

if __name__ == '__main__':
    settings.settings['visualize'] = False
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
   
    # models = [model(.4, .4, 180, 1), model(.4, .4, 180, 2), model(.4, .4, 180, 3)]
    # settingss = []
    # for j in range(3):
    #     settingss = settingss + [[i, i+1000, 1000, 1, 0.1, 0.2, 0.1, models[j]] for i in range(1000, 11000, 1000)]
    # processes = []
    # time_start = time.perf_counter()
    # #pool = Pool()
    # for i in settingss:
    #     a = Process(target=test_curves, args=[l, k] + i )
    #     a.start()
    #     processes.append(a)
    # for i in processes:
    #     i.join()
    # print(time.perf_counter() - time_start)
   
   

    test_resamplings = {'mult' : ParticleFilter.multinomial_resample}
    time_start = time.perf_counter()
    for i in range(1, 2):
        test_curves(1000, 1100, 1, 1, 0.1, 0.2, 0.1, model=model(.4, .4, 180, i))
    print(time.perf_counter() - time_start)

    #pool.close()
    #pool.join()
    # start_time = time.perf_counter()True
    #test_curves(2000, 2100, 100, 1, 0.1, 0.2, 0.1, modeTruel=model(.4, .4, 180, 2))
    #print(check_errors(1))
    # print(time.perf_counter() - start_time)
    # start_time = time.perf_counter()
    #test_models_new(2000, 2100, 100, 1, 0.5, 0.6, 0.1, model(.4, .4, 180, 1))
    # test_models_new(2000, 2100, 100, 1, 0.5, 0.6, 0.1, model(.4, .4, 180, .5))
    # print(time.perf_counter() - start_time)
    #print(time.perf_counter() - start_time)