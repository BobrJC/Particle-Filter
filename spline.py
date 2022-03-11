import copy
import math
import numpy as np

p = 2
n = 10
points = 1000

def numerical_interpolate(t, degree, points : np.array):
    i = s = l = 0
    n = len(points)

    assert degree > 1
    assert degree < (n-1)

    knots = [0]*(p+1) + [(i)/(n - degree) for i in range(1, n - p)] + [1]*(p+1)
    
    assert len(knots) == n + degree + 1
    
    domain = [degree, len(knots) - 1 - degree]
    
    for i in range(domain[0], domain[1]):
        if (t>=knots[i] and t <= knots[i+1]):
            s = i
            break
            
    v = copy.deepcopy(points)
        
    alpha = 0
    for l in range(1, degree + 2):
        for i in range(s, s-degree-1+l, -1):
            alpha = (t - knots[i]) / (knots[i+degree+1-l] - knots[i])
            v[i] = (1 - alpha) * v[i-1] + alpha * v[i]
    
    return v[s]


def create_curve(p, points, n=10000):
    ts = np.linspace(0, 1, n)
    curve = np.array([numerical_interpolate(t, p, points) for t in ts])
    
    return curve

def generate_points(n):
    return np.array([np.array([i, math.cos(i)]) for i in range(n)])


#Задаются точки для построения Би-сплайна


def get_curves():
    
    po = np.array([np.array([4., 3.]), np.array([5., 8.]), np.array([6., 2.]),
               np.array([7., 7.]), np.array([2., 7.]), np.array([7., 2.]), np.array([4., 2.])])
    curve = create_curve(p, po, points)

    eight = np.empty((2, 1000))
    with open('xs.out', 'r') as f:
        a = f.read().split('\n')
        x = list(map(float, a))

    with open('ys.out', 'r') as f:
        a = f.read().split('\n')
        y = list(map(float, a))
    eight[1] = x
    eight[1] -= 1
    eight[0] = y
    eight[0] += 1
    eight = np.rot90(eight, 3)

    
    return (curve, eight)
