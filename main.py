from random import uniform
import numpy as np

from Particle_filter import ParticleFilter
from Robot import Robot, model
from settings import settings
from tools import angle, dist, get_katets, vectors
from visualization import visualization
from datetime import datetime
import time
from spline import get_curves


BIG_ERRORS = []

def main(settings, curve = None):
    save = False
    if curve is not None:
        curve_len = 0
        for i in range(len(curve)):
            curve_len+=dist(curve[i], curve[i+1])
            if i == len(curve) - 2:
                break

        error = np.empty((settings['iterations'], 2))
        pf = ParticleFilter(settings['N_p'], settings['size_x'], settings['size_y'])
        a, b = vectors(curve[0], curve[1], curve[800])
        R = Robot(curve[1][0], curve[1][1], angle(a, b), settings['noize_rot'], settings['noize_dist'])
        R_coords = R.get_coords()
        R_angle = R.get_angle()
        mean = pf.estimate()
        part_start = pf.get_particles()

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

        particles_hist = {}
        mean_hist = {}
        Robot_hist = {}

        particles_hist.update({0:{'X':part_start[:,0], 'Y':part_start[:,1], 
                                    'Rotation':part_start[:,2]}})
        mean_hist.update({0:{'X':[mean[0]], 'Y':[mean[1]], 'Rotation':[mean[2]]}})
        Robot_hist.update({0: {'X':[R_coords[0]], 'Y':[R_coords[1]], 'Rotation':[R_angle]}})

        # if settings['fast'] == True:
        #     print('f', end= ' ')
        # else:
        #     print('s', end = ' ')
        
    
        for i in range(1, settings['iterations'] + 1):
            R.move_on_curve(curve, indexes[i], indexes[i-1])
            R_coords = R.get_coords()
            R_angle = R.get_angle()


            if settings['fast'] == True:
                a, b = vectors(curve[indexes[i-1]-1], curve[indexes[i-1]], curve[indexes[i]])
                c, d = vectors(curve[indexes[i-1]], curve[indexes[i]], curve[indexes[i]+1])
                resampled = pf.fast_start((dist(curve[indexes[i-1]], curve[indexes[i]]), angle(a, b), angle(c, d)), 
                                          R.get_coords(), settings['resample_type'], settings['noize_dist'], 
                                            settings['noize_rot'], settings['noize_sens'])
            else:
                a, b = vectors(curve[indexes[i-1]-1], curve[indexes[i-1]], curve[indexes[i]])
                pf.predict((dist(curve[indexes[i-1]], curve[indexes[i]]), angle(a, b)), settings['noize_dist'])

                a, b = vectors(curve[indexes[i-1]], curve[indexes[i]], curve[indexes[i]+1])
                pf.predict((0, angle(a, b)), 0)

                pf.weight(R_coords)
                if settings['resample_type'] == 'mult':
                    resampled = pf.multinomial_resample()
                elif settings['resample_type'] == 'strat':
                    resampled = pf.stratified_resample()
                elif settings['resample_type'] == 'syst':
                    resampled = pf.systematic_resample()

            mean = pf.estimate()
            
            error[i - 1] = np.asarray((dist(mean, R_coords), -(R_angle - mean[2])))
            if error[i-1][0] > .5:
                save = True
            particles_hist.update({i:{'X':resampled[:,0], 'Y':resampled[:,1], 
                                    'Rotation':resampled[:,2]}})
            mean_hist.update({i:{'X':[mean[0]], 'Y':[mean[1]], 'Rotation':[mean[2]]}})
            Robot_hist.update({i: {'X':[R_coords[0]], 'Y':[R_coords[1]], 'Rotation':[R_angle]}})
    else:
        R_model = model(0.4, 0.4, 120)
        R = Robot(settings['size_x'] / 2, settings['size_y'] / 2, uniform(0, 2*np.pi), settings['noize_rot'], settings['noize_dist'], model=R_model)
        pf = ParticleFilter(settings['N_p'], settings['size_x'], settings['size_y'])
        R_coords = R.get_coords()
        R_angle = R.get_angle()
        mean = pf.estimate()
        part_start = pf.get_particles()
        error = np.empty((settings['iterations'], 2))
        particles_hist = {}
        mean_hist = {}
        Robot_hist = {}
        particles_hist.update({0:{'X':part_start[:,0], 'Y':part_start[:,1], 
                                    'Rotation':part_start[:,2]}})
        mean_hist.update({0:{'X':[mean[0]], 'Y':[mean[1]], 'Rotation':[mean[2]]}})
        Robot_hist.update({0: {'X':[R_coords[0]], 'Y':[R_coords[1]], 'Rotation':[R_angle]}})
        
        
        for i in range(1, settings['iterations'] + 1):
            counter_of_tries = 0
            movement_angle = R.model_move(settings['size_x'], settings['size_y'])
            while movement_angle == None:
                movement_angle = R.model_move(settings['size_x'], settings['size_y'])
                counter_of_tries += 1
            R_coords = R.get_coords()
            R_angle = R.get_angle()
            
            if settings['fast'] == True:
                resampled = pf.fast_start((R.get_speed(), movement_angle, 0), R.get_coords(), 
                                            settings['resample_type'], settings['noize_dist'], 
                                            settings['noize_rot'], settings['noize_sens'])
            else:
                
                pf.predict((R.get_speed(), movement_angle), settings['noize_dist'])
                pf.weight(R_coords)
                if settings['resample_type'] == 'mult':
                    resampled = pf.multinomial_resample()
                elif settings['resample_type'] == 'strat':
                    resampled = pf.stratified_resample()
                elif settings['resample_type'] == 'syst':
                    resampled = pf.systematic_resample()
        
            mean = pf.estimate()
            #if i != 1:
            error[i - 1] = np.asarray((dist(mean, R_coords), -(R_angle - mean[2])))
            if error[i-1][0] > .5:
                print(1)
                save = True
            particles_hist.update({i:{'X':resampled[:,0], 'Y':resampled[:,1], 
                                    'Rotation':resampled[:,2]}})
            mean_hist.update({i:{'X':[mean[0]], 'Y':[mean[1]], 'Rotation':[mean[2]]}})
            Robot_hist.update({i: {'X':[R_coords[0]], 'Y':[R_coords[1]], 'Rotation':[R_angle]}})
    
    if settings['visualize']:
        visualization(Robot_hist, particles_hist, mean_hist, 
                    settings['size_x'], settings['size_y'], 
                    settings['N_p'], settings['iterations'], curve)
    if save:
        path = r"//home//bobr_js//Particle filter//"
        print('here')
        with open(path + 'BIG_ERRORS1', 'a') as f:
            f.write(str(settings)+str(error.tolist())+'\n')
    return error

if __name__ == '__main__':
    start_time = time.perf_counter()
    for i in range(1):
        main(settings, get_curves()[1])
    print(time.perf_counter() - start_time)