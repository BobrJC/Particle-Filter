
import numpy as np
#Размер поля


size_x = 10
size_y = 10
#Шумы
noize_dist = 0.2
noize_rot = 0.2
noize_sens = 0.1
#Ускоренный режим работы
fast = True
#Число итераций
iterations = 12
#Число частиц
N_p = 10000

fast = True
visualize = True

landmarks = np.array([[1., 1.], [1., 9.], [9., 1.], [9., 9.]])