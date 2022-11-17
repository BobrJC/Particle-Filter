import math
import numpy as np
from numpy.random import randn, random, uniform
from numpy.linalg import norm
from tools import angle, vectors
from settings import settings

# Класс модели поведения робота.
class model():
    # Конструктор.
    # Принимет: вероятность поворота налево, вероятность поворота направо, максимальный угол поворота в градусах, 
    # скорость робота. Тип входных знаений float или int.
    def __init__(self, prob_left, prob_right, max_angle, speed):
        if prob_left + prob_right > 1.:
            print('Wrong probability data')
        else:
            self.prob_left = prob_left
            self.prob_right = prob_right
            self.max_angle = max_angle*math.pi/180
            self.prob_forward = 1. - prob_left - prob_right
            self.speed = speed
    # Метод получения значения скорости.
    # Возвращяет значение скорости. Тип возвращаемого значения: float или int.
    def get_speed(self):
        return self.speed
    # Метод получения названия модели.
    # Возвращяет название модели. Тип возвращаемого значения: str.
    def get_name(self):
        return f'speed{self.speed}'
    # Метод определния следующего направления.
    # Возвращяет угол, на который повернется робот в радианах. Тип возвращяемого значения float.
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
    
# Класс робота
class Robot():
    # Конструктор.
    # Принимает: координаты начального положения по осям x и y, начальный угол поворота, 
    # шумы поворота и перемещения. Типы принимаемых значений float или int. Опционально 
    # принимает модель перемещения - объект класса model.
    def __init__(self, x, y, rot, noize_rot, noize_dist, model = None):
        self.x = x
        self.y = y
        self.rotation = rot
        self.model = model
        self.noize_rot = noize_rot
        self.noize_dist = noize_dist
    # Метод получения местоположения робота.
    # Возвращяет координаты по осям x и y. Тип возвращаемого значения: float.
    def get_coords(self):
        return self.x, self.y
    # Метод получения значения угла поворота.
    # Возвращяет значение угла поворота. Тип возвращаемого значения: float.
    def get_angle(self):
        return self.rotation
    # Метод получения значения скорости.
    # Возвращяет значение скорости. Тип возвращаемого значения: float или int.
    def get_speed(self):
        return self.model.get_speed()
    # Метод перемещения робота. Перемещает робота согласно принимаемым значениям.
    # Принимает список. Первое значение в списке - расстояние перемещения, второе - угол поворота.
    # Тип значений float или int.
    def move(self, movement):
        
        self.rotation += movement[1] + randn() * self.noize_rot
        self.rotation %= 2 * np.pi

        self.x += np.cos(self.rotation) * movement[0]
        self.y += np.sin(self.rotation) * movement[0]
        self.rotation += movement[2]
    # Метод перемещения согласно заданной модели. Перемещает робота с учетом размеров поля.
    # Принимает: размер поля по осям x и y.
    # Возвращает угол поворота или None, если робот выходит за рамки поля.
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
    # Метод перемешения по заданной кривой.
    # Принимаемые значения: список точек кривой, индекс точки в которой находится робот, индекс предыдущей точки. 
    # типы значений float для точек, int для индексов.
    def move_on_curve(self, curve, index, prev_index):
        a, b = vectors(curve[prev_index-1], curve[prev_index], curve[index])
        self.rotation += angle(a, b)
        a, b = vectors(curve[prev_index], curve[index], curve[index + 1])
        self.rotation += angle(a, b)
        self.x = curve[index][0]
        self.y = curve[index][1]