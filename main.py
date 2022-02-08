from random import uniform
import numpy as np

from Partucle_filter import ParticleFilter
from Robot import Robot, model
from settings import *
from tools import angle, dist, get_katets, vectors
from visualisation import visualize
from spline import curve

def main(resample_type, curve = None, visualization = False):
    if curve is not None:
        curve_len = 0
        for i in range(len(curve)):
            curve_len+=dist(curve[i], curve[i+1])
            if i == len(curve) - 2:
                break

        error = np.empty(iterations)
        pf = ParticleFilter(N_p, size_x, size_y)
        a, b = vectors(curve[0], curve[1], curve[800])
        R = Robot(curve[1][0], curve[1][1], angle(a, b))
        R_coords = R.get_coords()
        R_angle = R.get_angle()
        mean = pf.estimate()
        part_start = pf.get_particles()

        sum_len = 0
        h = curve_len / iterations
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

        if fast == True:
            print('f', end= ' ')
        else:
            print('s', end = ' ')
        
    
        for i in range(1, iterations + 1):
            R.move_on_curve(curve, indexes[i], indexes[i-1])
            R_coords = R.get_coords()
            R_angle = R.get_angle()


            if fast == True:
                a, b = vectors(curve[indexes[i-1]-1], curve[indexes[i-1]], curve[indexes[i]])
                c, d = vectors(curve[indexes[i-1]], curve[indexes[i]], curve[indexes[i]+1])
                resampled = pf.fast_start((dist(curve[indexes[i-1]], curve[indexes[i]]), angle(a, b), angle(c, d)), 
                                          R.get_dist_to_points(landmarks), resample_type)
            else:
                a, b = vectors(curve[indexes[i-1]-1], curve[indexes[i-1]], curve[indexes[i]])
                pf.predict((dist(curve[indexes[i-1]], curve[indexes[i]]), angle(a, b)), noize_dist)

                a, b = vectors(curve[indexes[i-1]], curve[indexes[i]], curve[indexes[i]+1])
                pf.predict((0, angle(a, b)), 0)

                pf.weight(R_coords)
                if resample_type == 'mult':
                    resampled = pf.multinomial_resample()
                elif resample_type == 'strat':
                    resampled = pf.stratified_resample()
                elif resample_type == 'syst':
                    resampled = pf.systematic_resample()

            mean = pf.estimate()
            if i != 1:
                error[i - 2] = dist(mean, R_coords)
            particles_hist.update({i:{'X':resampled[:,0], 'Y':resampled[:,1], 
                                    'Rotation':resampled[:,2]}})
            mean_hist.update({i:{'X':[mean[0]], 'Y':[mean[1]], 'Rotation':[mean[2]]}})
            Robot_hist.update({i: {'X':[R_coords[0]], 'Y':[R_coords[1]], 'Rotation':[R_angle]}})
    else:
        R_model = model(0.3, 0.5, 45)
        R = Robot(size_x / 2, size_y / 2, uniform(0, 2*np.pi), .5, R_model)
        pf = ParticleFilter(N_p, size_x, size_y)
        R_coords = R.get_coords()
        R_angle = R.get_angle()
        mean = pf.estimate()
        part_start = pf.get_particles()
        error = np.empty(iterations)
        particles_hist = {}
        mean_hist = {}
        Robot_hist = {}
        particles_hist.update({0:{'X':part_start[:,0], 'Y':part_start[:,1], 
                                    'Rotation':part_start[:,2]}})
        mean_hist.update({0:{'X':[mean[0]], 'Y':[mean[1]], 'Rotation':[mean[2]]}})
        Robot_hist.update({0: {'X':[R_coords[0]], 'Y':[R_coords[1]], 'Rotation':[R_angle]}})
        
        
        for i in range(1, iterations + 1):
            counter_of_tries = 0
            movement_angle = R.model_move()
            while movement_angle == None and counter_of_tries < 10:
                movement_angle = R.model_move()
                counter_of_tries += 1
            R_coords = R.get_coords()
            R_angle = R.get_angle()
            
            if fast == True:
                resampled = pf.fast_start((R.get_speed(), movement_angle, 0), R.get_dist_to_points(landmarks), resample_type)
            else:
                
                pf.predict((R.get_speed(), movement_angle), noize_dist)
                pf.weight(R_coords)
                #return pf.get_weights()
                if resample_type == 'mult':
                    resampled = pf.multinomial_resample()
                elif resample_type == 'strat':
                    resampled = pf.stratified_resample()
                elif resample_type == 'syst':
                    resampled = pf.systematic_resample()
        
            mean = pf.estimate()
            if i != 1:
                error[i - 2] = dist(mean, R_coords)
            particles_hist.update({i:{'X':resampled[:,0], 'Y':resampled[:,1], 
                                    'Rotation':resampled[:,2]}})
            mean_hist.update({i:{'X':[mean[0]], 'Y':[mean[1]], 'Rotation':[mean[2]]}})
            Robot_hist.update({i: {'X':[R_coords[0]], 'Y':[R_coords[1]], 'Rotation':[R_angle]}})
    
    if visualization:
        visualize(Robot_hist, particles_hist, mean_hist, curve)
    return error

if __name__ == '__main__':
    print(1)
    main('mult',visualization= visualize)