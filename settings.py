
import numpy as np
from Particle_filter import ParticleFilter
# Настройки фреймворка. Задаются режим работы фильтра (строка из списка modes), 
# тип используемой повторной выборки (метод класса ParticleFilter), 
# алгоритм адаптации числа частиц (строка из списка change_n), размеры поля по осям x и y (int или float), 
# шумы перемешения, поворота и измерений (float), число итераций (int), параметр детального вывода (bool), 
# параметр визуализации результатов (bool), параметр созранения больших значений ошибок (bool).
settings = {
    'mode' : 'fast',
    'resample_type' : ParticleFilter.multinomial_resample,
    'change_n' : 'KD',
    'fission': False,
    'size_x' : 10,
    'size_y' : 10,
    'N_p' : 5000,
    'noize_dist' : 0.2,
    'noize_rot' : 0.2,
    'noize_sens' : 0.2,
    'iterations' : 12,
    'detail': False,
    'visualize' : True,
    'save_errors': True,
    'save' : False
}
modes = ['normal', 'fast', 'change_n']
change_n = ['KD', 'KD_2', 'Pna']
#resamplings = ['syst', 'mult', 'strat']
test_resamplings = {'strat' : ParticleFilter.stratified_resample, 'syst' : ParticleFilter.systematic_resample, 'mult' : ParticleFilter.multinomial_resample}
curves = ['curve', 'eight']
