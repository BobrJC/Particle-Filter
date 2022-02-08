import math
import numpy as np
from numpy.random import randn, random, uniform
from numpy.linalg import norm
from tools import angle, vectors
from settings import *


class model():
    def __init__(self, prob_left, prob_right, max_angle):
        if prob_left + prob_right > 1.:
            print('Wrong probability data')
        else:
            self.prob_left = prob_left
            self.prob_right = prob_right
            self.max_angle = max_angle*math.pi/180
            self.prob_forward = 1. - prob_left - prob_right
    
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
    def __init__(self, x, y, rot, speed = None, model = None):
        self.x = x
        self.y = y
        self.rotation = rot
        self.speed = speed
        self.model = model
        
    def move(self, movement):
        
        self.rotation += movement[1] + randn() * noize_rot
        self.rotation %= 2 * np.pi

        self.x += np.cos(self.rotation) * movement[0]
        self.y += np.sin(self.rotation) * movement[0]
        self.rotation += movement[2]
        
    def model_move(self):
        angle = self.model.get_next_way()
        self.rotation += angle + randn() * noize_rot
        self.rotation %= 2 * np.pi
        
        self.x += np.cos(self.rotation) * self.speed
        self.y += np.sin(self.rotation) * self.speed
        if self.x > 0 and self.y > 0 and self.x < size_x and self.y < size_y:
            return angle
        else:
            self.x -= np.cos(self.rotation) * self.speed
            self.y -= np.sin(self.rotation) * self.speed
            return None
        
    def get_coords(self):
        return self.x, self.y
    
    def get_angle(self):
        return self.rotation
    
    def get_speed(self):
        return self.speed
    
    def move_on_curve(self, curve, index, prev_index):
        a, b = vectors(curve[prev_index-1], curve[prev_index], curve[index])
        self.rotation += angle(a, b)
        a, b = vectors(curve[prev_index], curve[index], curve[index + 1])
        self.rotation += angle(a, b)
        self.x = curve[index][0]
        self.y = curve[index][1]
    
    def get_dist_to_points(self, points):
        return (norm(points - np.array([self.x, self.y]), axis=1) + (randn(len(points)) * noize_sens))