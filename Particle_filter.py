from dataclasses import dataclass
import numpy as np
from numpy.random import randn, random, uniform
from sympy import im
from settings import *
from tools import dist
from scipy import stats
@dataclass
class Particle():
    x: float
    y: float
    rotation: float
    weight: float

class ParticleFilter():

    def __init__(self, N, x_lim, y_lim):
        self.particles = []
        self.N = N
        self.x_lim = x_lim
        self.y_lim = y_lim
        for i in range(N):
            self.particles.append(Particle(uniform(0, x_lim), uniform(0, y_lim), uniform(0, 2*np.pi), 1/N))
            
    def get_coords(self):
        res = np.empty((self.N, 2))
        for i in range(self.N):
            res[i][0] = self.particles[i].x
            res[i][1] = self.particles[i].y
        return res
    
    def get_weights(self):
        res = np.empty(self.N)
        for i in range(self.N):
            res[i] = self.particles[i].weight
        return res
    
    def get_particles(self):
        res = np.empty((self.N, 4))
        for i in range(self.N):
            res[i][0] = self.particles[i].x
            res[i][1] = self.particles[i].y
            res[i][2] = self.particles[i].rotation
            res[i][3] = self.particles[i].weight
        return res
   
    def set_particles(self, new_particles):
        for i in range(self.N):
            self.particles[i] = Particle(new_particles[i][0], new_particles[i][1], 
                                         new_particles[i][2], 1/self.N)
       
    def predict(self, u, noize_dist, noize_rot):
        for i in range(self.N):
            dist = u[0] + randn()*noize_dist
            
            self.particles[i].rotation += u[1] + randn()*noize_rot
            self.particles[i].rotation %= 2 * np.pi
            
            self.particles[i].x += np.cos(self.particles[i].rotation) * dist
            self.particles[i].y += np.sin(self.particles[i].rotation) * dist
            
    def weight(self, norm, noize_sens):
        sum_weight = 0
        distance_s = []
        distance0 = []
        for i in range(self.N):
            distance0.append(dist((self.particles[i].x, self.particles[i].y), (0,0)))
            distance = dist((self.particles[i].x, self.particles[i].y), norm)
            distance_s.append(distance0[i] + distance)
        norma = stats.norm(distance0, noize_sens).pdf(distance_s)
        for i in range(self.N):
            self.particles[i].weight *= norma[i]
            self.particles[i].weight += 1.e-12
            sum_weight += self.particles[i].weight
        for i in range(self.N):
            self.particles[i].weight/=sum_weight
            
    def neff(self):
        res = 0
        for i in range(self.N):
            res += self.particles[i].weight**.5
        return 1. / res
    
    def stratified_resample(self):
        if self.neff() < self.N/2:
            new_particles = np.empty((self.N, 3))
            positions = (random(self.N) + range(self.N)) / self.N
            cumulative_sum = [0]
            summ = 0
            for i in range(self.N):
                summ += self.particles[i].weight
                cumulative_sum.append(summ)
            i, j = 0, 0
            while i < self.N:
                if positions[i] < cumulative_sum[j]:
                    new_particles[i] = (self.particles[j - 1].x, self.particles[j - 1].y, 
                                        self.particles[j - 1].rotation)
                    i += 1
                else:
                    j += 1
            self.set_particles(new_particles)
            return new_particles
        else:
            return self.particles

    def systematic_resample(self):
        if self.neff() < self.N/2:
            new_particles = np.empty((self.N, 3))
            positions = (random() + np.arange(self.N)) / self.N
            cumulative_sum = [0]
            summ = 0
            for i in range(self.N):
                summ += self.particles[i].weight
                cumulative_sum.append(summ)
            i, j = 0, 0
            while i < self.N:
                if positions[i] < cumulative_sum[j]:
                    new_particles[i] = (self.particles[j - 1].x, self.particles[j - 1].y, 
                                        self.particles[j - 1].rotation)
                    i += 1
                else:
                    j += 1
            self.set_particles(new_particles)
            return new_particles
        else:
            return self.particles
            
    def multinomial_resample(self):
        cumulative_sum = [0]
        summ = 0
        for i in range(self.N - 1):
            summ += self.particles[i].weight
            cumulative_sum.append(summ)
        cumulative_sum[-1] = 1.
        indexes = np.searchsorted(cumulative_sum, random(self.N))
        old_particles = self.get_particles()[:,0:3]
        
        new_particles = np.empty((self.N, 3))
        new_particles[:] = old_particles[indexes - 1]
        self.set_particles(new_particles)
        return new_particles
    
    def fast_start(self, u, norm, resample_type, noize_dist, noize_rot, noize_sens):
        neff = 0
        sum_weight = 0
        distanc = 0
        distance = 0
        
        sum_dist = []
        cumulative_sum = [0]
        summ = 0
        distance0 = []
        for i in range(self.N):
            
            #prediction part
            distanc = u[0] + randn()*noize_dist
            #print(self.particles[i].rotation)
            self.particles[i].rotation += u[1] + randn()*noize_rot
            self.particles[i].rotation %= 2 * np.pi
            
            self.particles[i].x += np.cos(self.particles[i].rotation) * distanc
            self.particles[i].y += np.sin(self.particles[i].rotation) * distanc
            
            self.particles[i].rotation += u[2] + randn()*noize_rot
            self.particles[i].rotation %= 2 * np.pi
            #weight part
            distance0.append(dist((self.particles[i].x, self.particles[i].y), (0,0)))
            distance = dist((self.particles[i].x, self.particles[i].y), norm)
            sum_dist.append(distance0[i] + distance)
            #for j in range(len(landmarks)):
            #    distance[i][j] = dist((self.particles[i].x, self.particles[i].y), (landmarks[j]))

            
            #sum_dist.append(distance[i]+distance0[i])
        #particles = self.get_particles()
        # coeff = []
        # for i in range(len(norm)):
        #     coeff = stats.norm(distance[:,i], noize_sens).pdf(norm[i])
        #     for j in range(self.N):
        #         self.particles[j].weight *= coeff[j]
        #         self.particles[j].weight += 1.e-12
        # for j in range(self.N):
        #     sum_weight += self.particles[j].weight

        norma = stats.norm(distance0, noize_sens).pdf(sum_dist)
        for i in range(self.N):
           self.particles[i].weight *= norma[i]
           self.particles[i].weight += 1.e-12
           sum_weight += self.particles[i].weight
        #print(self.get_particles())
        
        for i in range(self.N):
            self.particles[i].weight/=sum_weight
            neff += self.particles[i].weight**.5
            #resampling part
            summ += self.particles[i].weight
            cumulative_sum.append(summ)
        neff = 1./neff
        if resample_type == 'mult':
            cumulative_sum[-1] = 1.
            indexes = np.searchsorted(cumulative_sum, random(self.N))
            old_particles = self.get_particles()[:,0:3]
            new_particles = np.empty((self.N, 3))
            new_particles[:] = old_particles[indexes - 1]
            self.set_particles(new_particles)
        elif resample_type == 'syst':
            new_particles = np.empty((self.N, 3))
            positions = (random() + np.arange(self.N)) / self.N
            i, j = 0, 0
            while i < self.N:
                if positions[i] < cumulative_sum[j]:
                    new_particles[i] = (self.particles[j - 1].x, self.particles[j - 1].y, 
                                        self.particles[j - 1].rotation)
                    i += 1
                else:
                    j += 1
            self.set_particles(new_particles)
            return new_particles
        elif resample_type == 'strat':
            new_particles = np.empty((self.N, 3))
            positions = (random(self.N) + range(self.N)) / self.N
            i, j = 0, 0
            while i < self.N:
                if positions[i] < cumulative_sum[j]:
                    new_particles[i] = (self.particles[j - 1].x, self.particles[j - 1].y, 
                                        self.particles[j - 1].rotation)
                    i += 1
                else:
                    j += 1
            self.set_particles(new_particles)
            return new_particles
        elif resample_type == 'resid':
            weights = self.get_weights()
            #cumulative_sum = [0]
            #summ = 0
            copies = self.N*np.asarray(weights).astype(int)
            k = 0
            indexes = np.zeros(self.N, 'i')
            for i in range(self.N):
                for _ in range(copies[i]): 
                    indexes[k] = i
                    k += 1
            residual = weights - copies
            #residual = copies
            #print(residual)
            residual /= np.sum(residual)
            cumulative_sum = np.cumsum(residual)
            cumulative_sum[-1] = 1.
            #for i in range(self.N):
            #    summ += residual[i]
            #    cumulative_sum.append(summ)
            indexes[k:self.N] = np.searchsorted(cumulative_sum, uniform(0, 1, self.N - k))
            old_particles = self.get_particles()[:,0:3]
            new_particles = np.empty((self.N, 3))
            new_particles[:] = old_particles[indexes - 1]
            self.set_particles(new_particles)
        return self.get_particles()
    
    def estimate(self):
        particles = self.get_particles()
        mean = np.average(particles[:,0:3], weights=particles[:, 3], axis=0)
        return mean
