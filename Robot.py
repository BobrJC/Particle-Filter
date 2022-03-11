import math
import numpy as np
from numpy.random import randn, random, uniform
from numpy.linalg import norm
from tools import angle, vectors
from settings import settings


class model():
    def __init__(self, prob_left, prob_right, max_angle, speed):
        if prob_left + prob_right > 1.:
            print('Wrong probability data')
        else:
            self.prob_left = prob_left
            self.prob_right = prob_right
            self.max_angle = max_angle*math.pi/180
            self.prob_forward = 1. - prob_left - prob_right
            self.speed = speed
    def get_speed(self):
        return self.speed
    def get_name(self):
        return f'speed{self.speed}'
    def get_next_way(self):
        way = np.searchsorted((self.prob_left, self.prob_right + self.prob_left), random())
        if way == 1:
            sgn = -1
        elif way == 0:
            sgn = 1
        else:
            sgn = 0
        angle = sgn*uniform(0, self.max_angle)
        return angle
    

class Robot():
    def __init__(self, x, y, rot, noize_rot, noize_dist, model = None):
        self.x = x
        self.y = y
        self.rotation = rot
        self.model = model
        self.noize_rot = noize_rot
        self.noize_dist = noize_dist
        
    def move(self, movement):
        
        self.rotation += movement[1] + randn() * self.noize_rot
        self.rotation %= 2 * np.pi

        self.x += np.cos(self.rotation) * movement[0]
        self.y += np.sin(self.rotation) * movement[0]
        self.rotation += movement[2]
        
    def model_move(self, size_x, size_y):
        angle = self.model.get_next_way()
        old_rot = self.rotation
        self.rotation += angle + randn() * self.noize_rot
        self.rotation %= 2 * np.pi
        
        self.x += np.cos(self.rotation) * self.model.get_speed()
        self.y += np.sin(self.rotation) * self.model.get_speed()
        if self.x > 0 and self.y > 0 and self.x < size_x and self.y < size_y:
            return angle
        else:
            self.x -= np.cos(self.rotation) * self.model.get_speed()
            self.y -= np.sin(self.rotation) * self.model.get_speed()
            self.rotation = old_rot
            return None
        
    def get_coords(self):
        return self.x, self.y
    
    def get_angle(self):
        return self.rotation
    
    def get_speed(self):
        return self.model.get_speed()
    
    def move_on_curve(self, curve, index, prev_index):
        a, b = vectors(curve[prev_index-1], curve[prev_index], curve[index])
        self.rotation += angle(a, b)
        a, b = vectors(curve[prev_index], curve[index], curve[index + 1])
        self.rotation += angle(a, b)
        self.x = curve[index][0]
        self.y = curve[index][1]
    def get_model_curve(self, n):
        coords = np.zeros(2)
        
        curve = np.zeros((n, 2))
        for i in range(n):
            curve[i]
    def get_dist_to_points(self, points, noize_sens):
        return (norm(points - np.array([self.x, self.y]), axis=1) + (randn(len(points)) * noize_sens))