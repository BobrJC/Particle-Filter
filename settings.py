
import numpy as np


settings = {
    'resample_type' : 'mult',
    'size_x' : 10,
    'size_y' : 10,
    'noize_dist' : 0.1,
    'noize_rot' : 0.1,
    'noize_sens' : 0.1,
    'fast' : True,
    'iterations' : 12,
    'N_p' : 5000,
    'fast' : True,
    'visualize' : True,
    'detail': False,
    'save_errors': True
}

resamplings = ['mult', 'strat', 'syst']
curves = ['curve', 'eight']

landmarks = np.array([[1., 1.], [1., 9.], [9., 1.], [9., 9.]])
