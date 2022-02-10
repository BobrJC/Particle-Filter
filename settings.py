
import numpy as np

settings = {
    'resample_type' : 'strat',
    'size_x' : 10,
    'size_y' : 10,
    #Шумы
    'noize_dist' : 0.05,
    'noize_rot' : 0.05,
    'noize_sens' : 0.05,
    #Ускоренный режим работы
    'fast' : True,
    #Число итераций
    'iterations' : 12,
    #Число частиц
    'N_p' : 1000,
    'fast' : True,
    'visualize' : False
}


landmarks = np.array([[1., 1.], [1., 9.], [9., 1.], [9., 9.]])