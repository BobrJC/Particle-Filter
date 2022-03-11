from random import uniform
import numpy as np

from Particle_filter import ParticleFilter
from Robot import Robot, model
from settings import settings
from tools import angle, dist, get_katets, vectors
from visualization import visualize_filter
from datetime import datetime
import time
from spline import get_curves
import os

def main(settings, curve = None, R_model = None):
    if settings['detail']:
        settings['fast'] = False
    error = []
    save = False
    particles_hist = []
    mean_hist = []
    Robot_hist = []

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

    

    particles_hist.append(part_start)
    mean_hist.append(mean)
    Robot_hist.append([R_coords, R_angle])
    error.append((dist(mean, R_coords), -(R_angle - mean[2])))
    
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


        if settings['fast'] == True:
            if curve is not None:
                a, b = vectors(curve[indexes[i-1]-1], curve[indexes[i-1]], curve[indexes[i]])
                c, d = vectors(curve[indexes[i-1]], curve[indexes[i]], curve[indexes[i]+1])
                resampled = pf.fast_start((dist(curve[indexes[i-1]], curve[indexes[i]]), angle(a, b), angle(c, d)), 
                                        R.get_coords(), settings['resample_type'], settings['noize_dist'], 
                                            settings['noize_rot'], settings['noize_sens'])
            else:
                resampled = pf.fast_start((R.get_speed(), movement_angle, 0), R.get_coords(), 
                                        settings['resample_type'], settings['noize_dist'], 
                                        settings['noize_rot'], settings['noize_sens'])
        else:
            if curve is not None:
                a, b = vectors(curve[indexes[i-1]-1], curve[indexes[i-1]], curve[indexes[i]])
                pf.predict((dist(curve[indexes[i-1]], curve[indexes[i]]), angle(a, b)), settings['noize_dist'], settings['noize_rot'])
                a, b = vectors(curve[indexes[i-1]], curve[indexes[i]], curve[indexes[i]+1])
                pf.predict((0, angle(a, b)), 0, 0)
            else:
                pf.predict((R.get_speed(), movement_angle), settings['noize_dist'], settings['noize_rot'])

            
            if settings['detail']:
                particles_hist.append(pf.get_particles()[:, 0:3])
                mean_hist.append(pf.estimate())
                Robot_hist.append([R_coords, R_angle])
                error.append((dist(mean, R_coords), -(R_angle - mean[2])))
            pf.weight(R_coords, settings['noize_sens'])
            if settings['resample_type'] == 'mult':
                resampled = pf.multinomial_resample()
            elif settings['resample_type'] == 'strat':
                resampled = pf.stratified_resample()
            elif settings['resample_type'] == 'syst':
                resampled = pf.systematic_resample()

        mean = pf.estimate()
        
        error.append((dist(mean, R_coords), -(R_angle - mean[2])))
        if error[i-1][0] > .5:
            save = True
        particles_hist.append(resampled)
        mean_hist.append(mean)
        Robot_hist.append([R_coords, R_angle])
    
    if settings['visualize']:
        visualize_filter(Robot_hist, particles_hist, mean_hist, 
                    settings['size_x'], settings['size_y'], 
                    settings['N_p'], settings['iterations'], error, curve)
    if save and settings['save_errors']:
        path = r"//home//bobr_js//Particle filter//"
        i = 1
        if os.path.getsize(path+f'BIG_ERRORS{i}') >= 1000000:
            i+=1
        with open(path + f'BIG_ERRORS{i}', 'a') as f:
            f.write(str(error) + str(settings['noize_dist']) + str(settings['resample_type']) +'\n')

    return error

if __name__ == '__main__':
    start_time = time.perf_counter()
    #for i in range(1):
    main(settings, R_model=model(.4, .4, 180, 3))
    # curve = get_curves()[0]
    print(time.perf_counter() - start_time)