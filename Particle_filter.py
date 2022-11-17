from dataclasses import dataclass
from random import randint
from types import new_class
import numpy as np
from numpy.random import randn, random, uniform, normal
from tools import dist, count_n, normpdf, angle, vectors
from scipy import stats

# Класс частицы. Содержит данные о положении, повороте частицы и её весе. 
@dataclass(unsafe_hash=True)
class Particle():
    x: float
    y: float
    rotation: float
    weight: float

# Класс фильтра частиц.
class ParticleFilter():
    # Конеструктор. 
    # Принимает число частиц (int), гланицы поля по осям x и y (float или int).
    def __init__(self, N, x_lim, y_lim):
        self.N = N
        self.particles = [Particle(uniform(0, x_lim), uniform(0, y_lim), uniform(0, 2*np.pi), 1/N) for part in range(self.N)]
        self.x_lim = x_lim
        self.y_lim = y_lim
    # Метод получения текущей ПРЧ. 
    # Возвращает текущую ПРЧ (float).
    def get_density(self):
        return self.N/(self.x_lim*self.y_lim)
    # Метод получения координат частиц. 
    # Возвращает numpy array с кооринатами частиц (float).
    def get_coords(self):
        res = np.empty((self.N, 2))
        for i in range(self.N):
            res[i][0] = self.particles[i].x
            res[i][1] = self.particles[i].y
        return res
    # Метод получения весов частиц. 
    # Возвращает numpy array с весами частиц (float).
    def get_weights(self):
        res = np.empty(self.N)
        for i in range(self.N):
            res[i] = self.particles[i].weight
        return res
    # Метод получения частиц. 
    # Возвращает numpy array с значениями полей частиц (float).
    def get_particles(self):
        res = np.empty((self.N, 4))
        for i in range(self.N):
            res[i][0] = self.particles[i].x
            res[i][1] = self.particles[i].y
            res[i][2] = self.particles[i].rotation
            res[i][3] = self.particles[i].weight
        return res
    # Метод установки частиц. 
    # Принимает список с значениями полей частиц.
    def set_particles(self, new_particles):
        self.particles = []
        if isinstance(new_particles, list):
            self.particles = new_particles
        else:
            for i in range(self.N):
                self.particles.append(Particle(new_particles[i][0], new_particles[i][1], 
                                            new_particles[i][2], 1/self.N))
    # Метод перемещения частиц согласно вектору перемещения. 
    # Принимает вектор перемещения с значением угла поворота
    # и длиной перемещения (float или int) и настройки фреймворка.
    def predict(self, u, settings):
        for i in range(self.N):
            dist = u[0] + randn()*settings['noize_dist']
            
            self.particles[i].rotation += u[1] + randn()*settings['noize_rot']
            self.particles[i].rotation %= 2 * np.pi
            
            self.particles[i].x += np.cos(self.particles[i].rotation) * dist
            self.particles[i].y += np.sin(self.particles[i].rotation) * dist

            self.particles[i].rotation += u[2] + randn()*settings['noize_rot']
            self.particles[i].rotation %= 2 * np.pi
    # Метод взвешивания чатсиц. 
    # Принимает положение робота (список с щначениями типа float) и настройки фреймворка.
    def weight(self, norm, settings):
        sum_weight = 0
        distance_s = []
        distance0 = []
        for i in range(self.N):
            distance0=dist((self.particles[i].x, self.particles[i].y), (0,0))
            distance = dist((self.particles[i].x, self.particles[i].y), norm)
            distance_s = distance0 + distance
            self.particles[i].weight *= normpdf(distance_s, distance0, settings)
            self.particles[i].weight += 1.e-12
            sum_weight += self.particles[i].weight
        for i in range(self.N):
            self.particles[i].weight/=sum_weight
    # Метод рассчета эффективного числа частиц. 
    # Возвращает жффективное число частиц (float).
    def neff(self):
        res = 0
        for i in range(self.N):
            res += self.particles[i].weight**.5
        return 1. / res
    # Метод стратифицированной повторной выборки. 
    # Принимает параметр возвращения частиц (float), число необходимых частиц (int), 
    # параметр установки отобранных частиц в список частиц фильтра (bool). 
    # Возвращает список отобранных частиц или их индексы или текущий набор частиц в фильтре.
    def stratified_resample(self, return_particles = True, N = None, set_part = True):
        if self.neff() < self.N/2:
            if N is None:
                N = self.N
            positions = (random(N) + range(N)) / N
            cumulative_sum = [0]
            summ = 0
            indexes = []
            new_particles = np.empty((N, 3))
            for i in range(self.N):
                summ += self.particles[i].weight
                cumulative_sum.append(summ)
            i, j = 0, 0
            while i < N:
                if positions[i] < cumulative_sum[j]:
                    indexes.append(j-1)
                    new_particles[i] = (self.particles[j - 1].x, self.particles[j - 1].y, 
                                        self.particles[j - 1].rotation)
                    i += 1
                else:
                    j += 1
            if set_part:
                self.set_particles(new_particles)
            if return_particles:
                return new_particles
            else:
                return indexes
        else:
            return self.particles
    # Метод систематической повторной выборки. 
    # Принимает параметр возвращения частиц (float), число необходимых частиц (int), 
    # параметр установки отобранных частиц в список частиц фильтра (bool). 
    # Возвращает список отобранных частиц или их индексы или текущий набор частиц в фильтре.
    def systematic_resample(self, return_particles = True, N = None, set_part = True):
        if self.neff() < self.N/2:
            if N is None:
                N = self.N
            new_particles = np.empty((N, 3))
            positions = (random() + np.arange(N)) / N
            cumulative_sum = [0]
            indexes = []
            summ = 0
            for i in range(self.N):
                summ += self.particles[i].weight
                cumulative_sum.append(summ)
            i, j = 0, 0
            while i < N:
                if positions[i] < cumulative_sum[j]:
                    indexes.append(j - 1)
                    new_particles[i] = (self.particles[j - 1].x, self.particles[j - 1].y, 
                                        self.particles[j - 1].rotation)
                    i += 1
                else:
                    j += 1
            if set_part:
                self.set_particles(new_particles)
            if return_particles:
                return new_particles
            else:
                return indexes
        else:
            return self.particles
    # Метод полиномиальной повторной выборки. 
    # Принимает параметр возвращения частиц (float), число необходимых частиц (int), 
    # параметр установки отобранных частиц в список частиц фильтра (bool). 
    # Возвращает список отобранных частиц или их индексы или текущий набор частиц в фильтре.
    def multinomial_resample(self, return_particles = True, N = None, set_part = True):
        if N is None:
            N = self.N
        cumulative_sum = [0]
        summ = 0
        particles = self.get_particles()
        old_particles = particles[:,0:3]
        weights = particles[:, 3]
        for i in range(self.N - 1):
            summ += weights[i]
            cumulative_sum.append(summ)
        cumulative_sum[-1] = 1.
        indexes = np.searchsorted(cumulative_sum, random(N))
        new_particles = np.empty((N, 3))
        new_particles[:] = old_particles[indexes - 1]
        if set_part:
            self.set_particles(new_particles)
        if return_particles:
            return new_particles
        else:
            return indexes
    # Метод расщепляющей повторной выброрки. 
    # Принимает положение робота, настройки фреймворка. 
    # Опционально принимает число частиц (int).
    # Возвращает список новых частиц.
    def fissial_resample_2(self, norm, settings, N = None):
        if N is None:
            N = self.N
        np.seterr('raise')
        list_particles = [[],[],[],[]]
        new_particles = []
        i = 0
        weights = self.get_weights()
        max_weight = max(weights)
        ave_weights = 1/N
        sum_weight = 0
        good_weights = 0
        for i in range(N):
        
            good_weights += 1
            fission_factor = 1/(1+np.exp((self.particles[i].weight - ave_weights)/(max_weight - ave_weights)))
            fission_len = round(N*self.particles[i].weight)+2
            list_particles[0] = list_particles[0] + list(normal(self.particles[i].x, fission_factor, fission_len))
            list_particles[1] = list_particles[1] + list(normal(self.particles[i].y, fission_factor, fission_len))
            list_particles[2] = list_particles[2] + [self.particles[i].rotation]*fission_len
            list_particles[3] = list_particles[3] + [self.particles[i].weight]*fission_len
        list_particles = np.asarray(list_particles).transpose()
        for i in range(len(list_particles)):
            distance0=dist((list_particles[i][0], list_particles[i][1]), (0,0))
            distance = dist((list_particles[i][0], list_particles[i][1]), norm)
            distance_s = distance0 + distance
            list_particles[i][3] *= round(normpdf(distance_s, distance0, settings), 8)
            list_particles[i][3] += 1.e-12
        list_particles = np.asarray(sorted(list_particles, reverse = True, key=lambda x: x[3])[0:N])
        sum_weight = sum(list_particles[:,3])
        self.particles = []            
        for i in range(N):
            list_particles[i][3]/=sum_weight
            self.particles.append(Particle(*list_particles[i]))
        return new_particles
        
    # Метож секторного РКЛ алгоритма. 
    # Принимет необходимую точность (float), угол сектора (float),  
    # минимально число чатсиц (int), вектор перемешения (список с данными типа float), 
    # старое и новое положение робота (списки с данными типа float), настройки фреймворка.
    def KL_dist_recount_n(self, eps, bin_size, min_n, u, new_norm, old_norm, settings):
        bins_untouched = [True]*(360//bin_size)
        new_particles = []
        new_n = 0
        n_x = 0
        k = 0
        sum_weight = 0
        indexes = []
        print(settings, "sett KL")
        indexes = settings['resample_type'](self, return_particles=False)
        
        while True:
            index = randint(0, len(indexes) - 1)
            particle = Particle(self.particles[indexes[index]].x, self.particles[indexes[index]].y,
                                 self.particles[indexes[index]].rotation, self.particles[indexes[index]].weight)
            move_dist = u[0] + randn()*settings['noize_dist']
            
            particle.rotation += u[1] + randn()*settings['noize_rot']
            particle.rotation %= 2 * np.pi
            
            particle.x += np.cos(particle.rotation) * move_dist
            particle.y += np.sin(particle.rotation) * move_dist

            particle.rotation += u[2] + randn()*settings['noize_rot']
            particle.rotation %= 2 * np.pi

            dist0 = dist((particle.x, particle.y), (0,0))
            distance = dist((particle.x, particle.y), new_norm)
            particle.weight *= normpdf(dist0+distance,  dist0, settings)
            particle.weight += 1.e-12
            sum_weight += particle.weight
            new_particles.append(particle)
            vec = vectors((old_norm[0] + 1, old_norm[1]), old_norm, (particle.x, particle.y))
            bin_angle = angle(vec[0], vec[1])
            if bin_angle < 0:
                index = int((-bin_angle*180/np.pi) // bin_size)
            else:
                index = int((360 - bin_angle*180/np.pi) // bin_size)
            if index >= len(bins_untouched):
                index = len(bins_untouched) - 1
            if bins_untouched[index] and dist((particle.x, particle.y), old_norm) < u[0] + settings['noize_dist']:
                k += 1
                bins_untouched[index] = False
                if k>1:
                    n_x = count_n(eps, k)
            new_n += 1
            if new_n > min_n and new_n > n_x:
                break
        self.N = new_n
        for i in range(self.N):
            new_particles[i].weight/=sum_weight
        self.set_particles(new_particles)
        if settings['fission']:
            self.fissial_resample_2(new_norm, settings)
        return self.get_particles()

    # Метод РКЛ выборки с двумя наборами частиц. 
    # Принимает вектор перемешения (список с данными типа float), 
    # положение робота (список с данными типа float), коэффициент уменьшения выборки (float < 1),
    # коэффициент увеличения выборки (float > 1), коэффициент числа частиц в меньшем наборе (float < alpha), настройки фреймворка.
    def KL_dist_recount_two_rows(self, u, norm, alpha, betta, c, settings):
        N = round(self.N/c)
        KLdist = 0
        self.predict(u, settings)
        self.weight(norm, settings)
        if settings['fission'] and settings['resample_type'] != ParticleFilter.multinomial_resample:
            self.fissial_resample_2(norm, settings)
            eps = 0.15
        elif settings['fission'] and settings['resample_type'] == ParticleFilter.multinomial_resample:
            eps = 1.3
        elif settings['fission'] == False and settings['resample_type'] == ParticleFilter.multinomial_resample:
            eps = 1.3
        else:
            eps = 0.188
        indexes_low = list(settings['resample_type'](self, N= N, set_part= False, return_particles=False))
        prob_dist_low = []
        sum_prob_low = 0
        prob_dist_high = []
        sum_prob_high = 0
        for i in set(indexes_low):
            count = indexes_low.count(i)
            prob_dist_high.append(self.particles[i].weight)
            prob_dist_low.append(count)
            sum_prob_low+=count
            sum_prob_high+= self.particles[i].weight
        prob_dist_low = list(map(lambda x: x/sum_prob_low, prob_dist_low))
        prob_dist_high = list(map(lambda x: x/sum_prob_high, prob_dist_high))
        for i, index in enumerate(set(indexes_low)):
            KLdist += prob_dist_high[i]*np.log2(prob_dist_high[i]/prob_dist_low[i])
        
        if KLdist < eps:
            N_new = round(self.N*betta)
        else:
            N_new = round(self.N*alpha)
        if N_new < 150:
            N_new = 150
        elif N_new > settings['N_p']*1.5:
            N_new = settings['N_p']
        particles = settings['resample_type'](self, N= N_new, set_part= False)
        self.N = N_new
        self.set_particles(particles)
        return particles
    # Метод АВЧ по апостериорному распределению. 
    # Принимает вектор перемешения (список с данными типа float), 
    # положение робота (список с данными типа float), нижнюю границу диапозона числа частиц (int), 
    # верхнюю границу диапозона числа частиц (int), вероятностный интервал (float), 
    # коэффициент уменьшения выборки (float < 1), коэффициент увеличения выборки (float > 1), настройки фреймворка.
    def Pmabopd(self, u, norm, N1, N2, theta, a, b, settings):

        self.predict(u, settings)
        self.weight(norm, settings)
        if settings['fission']:
            self.fissial_resample_2(norm, settings)
        N_theta = 0
        for i in range(self.N):
            if self.particles[i].weight < 1 - theta:
                N_theta += 1
        if N_theta > N1:
            N = round(a*self.N)
        elif N_theta < N2:
            N = round(b*self.N)
        
        particles = settings['resample_type'](self, N= N, set_part= False)
        self.N = N
        self.set_particles(particles)
       
        return particles


    # Метод ускоренного режима работы фильтра. Не работает при адаптации числа чатсиц.
    # Принимает вектор перемешения (список с данными типа float), 
    # положение робота (список с данными типа float), настройки фреймворка.
    def fast_start(self, u, norm, settings):
        neff = 0
        sum_weight = 0
        distanc = 0
        distance = 0
        
        sum_dist = []
        cumulative_sum = [0]
        summ = 0
        distance0 = []
        for i in range(self.N):
            
            distanc = u[0] + randn()*settings['noize_dist']
            self.particles[i].rotation += u[1] + randn()*settings['noize_rot']
            self.particles[i].rotation %= 2 * np.pi
            
            self.particles[i].x += np.cos(self.particles[i].rotation) * distanc
            self.particles[i].y += np.sin(self.particles[i].rotation) * distanc
            
            self.particles[i].rotation += u[2] + randn()*settings['noize_rot']
            self.particles[i].rotation %= 2 * np.pi

            distance0.append(dist((self.particles[i].x, self.particles[i].y), (0,0)))
            distance = dist((self.particles[i].x, self.particles[i].y), norm)
            sum_dist.append(distance0[i] + distance)

        norma = stats.norm(distance0, settings['noize_sens']).pdf(sum_dist)
        for i in range(self.N):
           self.particles[i].weight *= norma[i]
           self.particles[i].weight += 1.e-12
           sum_weight += self.particles[i].weight
        new_weights = []
        for i in range(self.N):
            self.particles[i].weight/=sum_weight
            new_weights.append(self.particles[i].weight)
            neff += self.particles[i].weight**2
            summ += self.particles[i].weight
            cumulative_sum.append(summ)
        neff = 1./neff
        if settings['resample_type'] == ParticleFilter.multinomial_resample:
            cumulative_sum[-1] = 1.
            indexes = np.searchsorted(cumulative_sum, random(self.N))
            old_particles = self.get_particles()[:,0:3]
            new_particles = np.empty((self.N, 3))
            new_particles[:] = old_particles[indexes - 1]
            self.set_particles(new_particles)
        elif settings['resample_type'] == ParticleFilter.systematic_resample:
            new_particles = np.empty((self.N, 3))
            positions = (random() + np.arange(self.N)) / self.N
            i, j = 0, 0
            while i < self.N:
                if positions[i] < cumulative_sum[j]:
                    new_particles[i] = (self.particles[j - 1].x, self.particles[j - 1].y, 
                                        self.particles[j - 1].rotation)
                    i += 1
                else:
                    j += 1
            self.set_particles(new_particles)
            return new_particles
        elif settings['resample_type'] == ParticleFilter.stratified_resample:
            new_particles = np.empty((self.N, 3))
            positions = (random(self.N) + range(self.N)) / self.N
            i, j = 0, 0
            
            while i < self.N:
                if positions[i] < cumulative_sum[j]:
                    new_particles[i] = (self.particles[j - 1].x, self.particles[j - 1].y, 
                                        self.particles[j - 1].rotation)
                    i += 1
                else:
                    j += 1
            self.set_particles(new_particles)
            return new_particles
        elif settings['resample_type'] == 'resid':
            weights = self.get_weights()
            copies = self.N*np.asarray(weights).astype(int)
            k = 0
            indexes = np.zeros(self.N, 'i')
            for i in range(self.N):
                for _ in range(copies[i]): 
                    indexes[k] = i
                    k += 1
            residual = weights - copies
            residual /= np.sum(residual)
            cumulative_sum = np.cumsum(residual)
            cumulative_sum[-1] = 1.
            indexes[k:self.N] = np.searchsorted(cumulative_sum, uniform(0, 1, self.N - k))
            old_particles = self.get_particles()[:,0:3]
            new_particles = np.empty((self.N, 3))
            new_particles[:] = old_particles[indexes - 1]
            self.set_particles(new_particles)
        return self.get_particles()
    # Метод получения усредненного положения частиц. 
    # Возвращает усредненное положение частиц (список с данными типа float).
    def estimate(self):
        particles = self.get_particles()
        mean = np.average(particles[:,0:3], weights=particles[:, 3], axis=0)
        return mean
