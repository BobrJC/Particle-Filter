import math

def angle(a, b):
    sgn = 1
    if a[0]*b[1] - a[1]*b[0] < 0:
        sgn = -1
    return sgn*math.acos((a[0]*b[0] + a[1]*b[1])/(math.sqrt(math.pow(a[0], 2) + math.pow(a[1], 2))*math.sqrt(math.pow(b[0],2) + math.pow(b[1], 2))))

def dist(a, b):
    return ((b[1]-a[1])**2 + (b[0] - a[0])**2)**.5

def get_katets(a, b, length):
    kat1 = b[0] - a[0]
    kat2 = b[1] - a[1]
    koeff = dist(a, b) / length
    return (kat1 / koeff, kat2 / koeff)

def vectors(a, b, c):
    return (b[0] - a[0], b[1] - a[1]), (c[0] - b[0], c[1] - b[1])