from random import uniform
import numpy as np

from Particle_filter import ParticleFilter
from Robot import Robot, model
import settings
from tools import angle, dist, get_katets, get_script_dir, vectors
from visualization import visualize_filter
import time
from spline import get_curves
import os

# Функция запуска фильтра частиц. 
# Принимает настройки фреймворка. 
# Опционально принимает используемую кривую (список с данными типа float) 
# или модель пермещения робота (обхект класса model). 
# Возвращает список с ошибками (список с данными типа float) или список 
# с ошибками и список с историей изменения ПРЧ (список с данными типа float).
def main(settings, curve = None, R_model = None):
    print("main")
    if settings['detail']:
        settings['fast'] = False
    
    error = []
    save = False
    particles_hist = []
    mean_hist = []
    Robot_hist = []
    Density_hist = []

    if curve is not None:
        curve_len = 0
        for i in range(len(curve)):
            curve_len+=dist(curve[i], curve[i+1])
            if i == len(curve) - 2:
                break

        a, b = vectors(curve[0], curve[1], curve[800])
        R = Robot(curve[1][0], curve[1][1], angle(a, b), 
                    settings['noize_rot'], settings['noize_dist'])
        sum_len = 0
        h = curve_len / settings['iterations']
        indexes = [0]
        for i in range(len(curve) - 1):
            sum_len += dist(curve[i], curve[i+1])
            if sum_len > h:
                if sum_len > h + .1:
                    katets = get_katets(curve[i], curve[i+1], dist(curve[i], curve[i+1]))
                    curve = np.insert(curve, i, np.array([curve[i][0] + katets[0], curve[i][1] + katets[1]]), axis = 0)
                indexes.append(i)
                sum_len = 0
        indexes.append(len(curve) - 2)
        indexes.append(len(curve) - 1)
    else:
        R = Robot(settings['size_x'] / 2, settings['size_y'] / 2, uniform(0, 2*np.pi), settings['noize_rot'], settings['noize_dist'], model=R_model)
        
    pf = ParticleFilter(settings['N_p'], settings['size_x'], settings['size_y'])
    R_coords = R.get_coords()
    R_angle = R.get_angle()
    mean = pf.estimate()
    part_start = pf.get_particles()
    density = pf.get_density()
    if settings['change_n'] == 'KD':
        pf.weight(R_coords, settings)
    particles_hist.append(part_start)
    mean_hist.append(mean)
    Robot_hist.append([R_coords, R_angle])
    error.append((dist(mean, R_coords)))
    Density_hist.append(density)

    for i in range(1, settings['iterations'] + 1):
        if curve is not None:
            R.move_on_curve(curve, indexes[i], indexes[i-1])
        else:
            counter_of_tries = 0
            movement_angle = R.model_move(settings['size_x'], settings['size_y'])
            while movement_angle == None:
                movement_angle = R.model_move(settings['size_x'], settings['size_y'])
                counter_of_tries += 1
        R_coords = R.get_coords()
        R_angle = R.get_angle()

        if curve is not None:
            a, b = vectors(curve[indexes[i-1]-1], curve[indexes[i-1]], curve[indexes[i]])
            c, d = vectors(curve[indexes[i-1]], curve[indexes[i]], curve[indexes[i]+1])
            u = (dist(curve[indexes[i-1]], curve[indexes[i]]), angle(a, b), angle(c, d))
        else:
            u = (R.get_speed(), movement_angle, 0)
        if settings['mode'] == 'change_n':
            if settings['change_n'] == 'KD':
                eps = 0.05
                bin = 10
                if i == 1:
                    resampled = pf.KL_dist_recount_n(eps, bin, 100, u, R.get_coords(), Robot_hist[i - 1][0], settings)

                else:
                    resampled = pf.KL_dist_recount_n(eps, bin, 100, u, R.get_coords(), Robot_hist[i - 1][0], settings)
            elif settings['change_n'] == 'KD_2':
                if settings['resample_type'] == ParticleFilter.systematic_resample:
                    c = 3
                else:
                    c = 2.5
                if i == 1:
                    c = 1
                resampled = pf.KL_dist_recount_two_rows(u, R.get_coords(), 1.3, 0.7, c, settings)
            elif settings['change_n'] == 'Pna':
                resampled = pf.Pmabopd(u, R.get_coords(), 100, 200, .99, .75, 1.5, settings)
        elif settings['mode'] == 'fast':
            resampled = pf.fast_start(u, R.get_coords(), settings)
        elif settings['mode'] == 'normal':
            
            pf.predict(u, settings)

            if settings['detail']:
                particles_hist.append(pf.get_particles()[:, 0:3])
                mean_hist.append(pf.estimate())
                Robot_hist.append([R_coords, R_angle])
                error.append((dist(mean, R_coords)))
                Density_hist.append(pf.get_density())
            pf.weight(R_coords, settings)
            resampled = settings['resample_type'](pf)
 
        mean = pf.estimate()
        
        error.append((dist(mean, R_coords)))
        if error[i-1] > .3:
            save = True
        particles_hist.append(resampled)
        mean_hist.append(mean)
        Robot_hist.append([R_coords, R_angle])
        Density_hist.append(pf.get_density())
    
    if settings['visualize']:
        visualize_filter(Robot_hist, particles_hist, mean_hist, Density_hist,
                    settings['size_x'], settings['size_y'], 
                    settings['iterations'], error, curve)
    if save and settings['save_errors']:
        path = get_script_dir()
        i = 1
        while os.path.getsize(path+f'/BIG_ERRORS{i}')  >= 10000000 and os.path.exists(path+f'/BIG_ERRORS{i}'):
            i+=1
        with open(path + f'BIG_ERRORS{i}', 'a') as f:
            f.write(str(error) + str(settings['noize_dist']) + str(settings['resample_type']) +str(settings['change_n']) + '\n')
    
    if settings['change_n'] == None:
        return error
    else:
        return error, Density_hist

if __name__ == '__main__':
    curve = get_curves()[1]
    start_time = time.perf_counter()
    error = []
    n = 1
    for i in range(n):
        error.append(main(settings.settings, curve))# R_model=model(.4, .4, 180, 1)))