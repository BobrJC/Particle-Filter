import math
import os
import sys
import inspect

def angle(a, b):
    sgn = 1
    if a[0]*b[1] - a[1]*b[0] < 0:
        sgn = -1
    part = math.sqrt(math.pow(a[0], 2) + math.pow(a[1], 2))*math.sqrt(math.pow(b[0],2) + math.pow(b[1], 2))
    if part == 0:
        return 0
    return sgn*math.acos((a[0]*b[0] + a[1]*b[1])/part)

def dist(a, b):
    return ((b[1]-a[1])**2 + (b[0] - a[0])**2)**.5

def get_katets(a, b, length):
    kat1 = b[0] - a[0]
    kat2 = b[1] - a[1]
    koeff = dist(a, b) / length
    return (kat1 / koeff, kat2 / koeff)

def vectors(a, b, c):
    return (b[0] - a[0], b[1] - a[1]), (c[0] - b[0], c[1] - b[1])


def get_script_dir(follow_symlinks=True):
    if getattr(sys, 'frozen', False):
        path = os.path.abspath(sys.executable)
    else:
        path = inspect.getabsfile(get_script_dir)
    if follow_symlinks:
        path = os.path.realpath(path)
    return os.path.dirname(path)

def update_folder(folder_path):
    script_dir = get_script_dir()
    if not os.path.exists(script_dir + folder_path):
        os.mkdir(script_dir + folder_path)

def count_n(eps, k):
    return ((k-1)/(2*eps))*((1 - 2/(9*(k-1)) + (2/(9*(k-1)))**.5*2.326)**3)

def normpdf(x, mean, settings):
    var = float(settings['noize_sens'])**2
    denom = (2*math.pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom