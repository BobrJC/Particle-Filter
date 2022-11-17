
from bokeh.plotting import figure, output_file, save
from bokeh.io import show
from bokeh.models import ColumnDataSource, CustomJS, Slider
from bokeh.layouts import column
import settings
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tools import get_script_dir
from numpy import arange, asarray
import numpy as np

# Функция виуализации результатов работы фильтра частиц. 
# Принимает историю перемешения робота, историю перемещения частиц, 
# историю изменения медианного положения частиц, историю изменения ПРЧ 
# (списки со значениями типа float), размер поля по осям x и y (float), число итераций (int), 
# историю изменения медианной ошибки (список со значениями типа float). Опционально принимает используемую кривую.
def visualize_filter(robot_hist, particles_hist, mean_hist, density_hist, size_x, size_y, iterations, error_hist, curve = None, settings = settings.settings):
    Robot_data = {}
    particles_data = {}
    mean_data = {}
    error_hist = asarray(error_hist)
    error_data = {}
    density_data = {}
    for i, data in enumerate(robot_hist):
        Robot_data.update({i : {'X' : [data[0][0]], 'Y' : [data[0][1]], 'Rotation' : [data[1]]}})
    for i, data in enumerate(particles_hist):
        particles_data.update({i : {'X' : data[:,0], 'Y' : data[:,1], 'Rotation' : data[:,2]}})
    for i, data in enumerate(mean_hist):
        mean_data.update({i : {'X' : [data[0]], 'Y' : [data[1]], 'Rotation' : [data[2]]}})
    for i, data in enumerate(error_hist):
        error_data.update({i : {'Error' : [f'Ошибка: {round(data, 5)}']}})
    for i, data in enumerate(density_hist):
        density_data.update({i : {'Density' : [f'ПРЧ: {round(data, 2)}']}})
    

    pl = figure(plot_width = 800, plot_height = 800, 
                    x_range = (0, size_x), y_range = (0, size_y), 
        )
    source_visible_particles = ColumnDataSource(particles_data[0])
    source_visible_robot = ColumnDataSource(Robot_data[0])
    source_visible_mean = ColumnDataSource(mean_data[0])
    source_visible_error = ColumnDataSource(error_data[0])
    source_visible_density = ColumnDataSource(density_data[0])

    pl.triangle(
        'X', 'Y', angle = 'Rotation',
        size = 10, 
        fill_color = "violet",
        line_color = "violet",
        fill_alpha = 0.10,
        line_width = 1,
        source = source_visible_particles
      )
    pl.triangle_dot(
        'X', 'Y', angle = 'Rotation',
        size = 30,
        fill_color = 'green',
        line_color = 'green',
        legend_label = "Робот",
        source = source_visible_robot
    )
    pl.triangle_dot(
        'X', 'Y', angle = 'Rotation',
        size = 30,
        fill_color = 'red',
        line_color = 'red',
        legend_label = "Усредненное положение частиц",
        source = source_visible_mean
    )
    pl.circle(
        source = source_visible_error,
        x = 0, y = 0, size=0, 
        legend='Error'
    )
    pl.circle(
        source = source_visible_density,
        x = 0, y = 0, size=0, 
        legend='Density'
    )
    
    if curve is not None:
        pl.line(x = curve[:,0], y = curve[:,1], line_width = 3, legend='Траектория')
    else:
        xs = list(map(lambda item: item['X'], Robot_data.values()))
        ys = list(map(lambda item: item['Y'], Robot_data.values()))
        pl.line(x = xs, y = ys, line_width = 5, legend='Траектория')

    if not settings['detail']:
        slider = Slider(start=0, end=iterations, value=0, step=1, title="Iteration")
    else:
        slider = Slider(start=0, end=2*iterations, value=0, step=1, title="Stage")
    callback = CustomJS(args=dict(source_particles=particles_data, 
                                  source_robot = Robot_data,
                                  source_mean = mean_data,
                                  source_error = error_data,
                                  source_density = density_data,
                                  source_visible_particles = source_visible_particles,
                                  source_visible_robot = source_visible_robot,
                                  source_visible_mean = source_visible_mean,
                                  source_visible_error = source_visible_error,
                                  source_visible_density = source_visible_density,
                                  iteration=slider), 
                                    code="""
    var iter = iteration.value;
    var data_visible_particles = source_visible_particles.data;
    var data_availabel_particles = source_particles.data;
    data_visible_particles.X = source_particles[iter].X
    data_visible_particles.Y = source_particles[iter].Y
    data_visible_particles.Rotation = source_particles[iter].Rotation
    source_visible_particles.change.emit();
    
    var data_visible_robot = source_visible_robot.data;
    var data_availabel_robot = source_robot.data;
    data_visible_robot.X = source_robot[iter].X
    data_visible_robot.Y = source_robot[iter].Y
    data_visible_robot.Rotation = source_robot[iter].Rotation
    source_visible_robot.change.emit();
    
    var data_visible_mean = source_visible_mean.data;
    var data_availabel_mean = source_mean.data;
    data_visible_mean.X = source_mean[iter].X
    data_visible_mean.Y = source_mean[iter].Y
    data_visible_mean.Rotation = source_mean[iter].Rotation
    source_visible_mean.change.emit();

    var data_visible_error = source_visible_error.data;
    var data_availabel_error = source_error.data;
    data_visible_error.Error = source_error[iter].Error
    source_visible_error.change.emit();

    var data_visible_density = source_visible_density.data;
    var data_availabel_density = source_density.data;
    data_visible_density.Density = source_density[iter].Density
    source_visible_Density.change.emit();
    """)
    
    pl.xaxis.axis_label = 'U'
    pl.yaxis.axis_label = 'K'
    pl.xaxis.major_label_text_font_size = '23pt'
    pl.yaxis.major_label_text_font_size = '23pt'

    pl.xaxis.axis_label_text_font_size = '30pt'
    pl.yaxis.axis_label_text_font_size = '30pt'
    pl.xaxis.axis_label_text_color = 'black'
    pl.yaxis.axis_label_text_color = 'black'
    pl.legend.label_text_font_size = '22pt'
    pl.legend.label_text_color='black'
    slider.js_on_change('value', callback)
    layout = column(slider, pl)
    output_file(filename="PF_KDA_DIST.html", title="PF_KDA_DIST")
    show(layout)

# Функция визуализации изменения медианной ошибки. 
# Принимает число повторений тестирования. 
# Опционально принимает типы повторной выборки, шумы (список с данными типа float), 
# начальное число частиц (список с данными типа int), пути к файлам с историей 
# медианной ошибки (список с данными типа str), список необходимых данных в легенде (список с данными типа str), 
# используемые кривые (список с данными типа str), параметр использования расщепляющей повторной вбыорки (bool), 
# параметр отображения изменений ПРЧ (bool), имя файла в который будет сохранен чертеж (str).

def visualize_errors(N, resamplings = list(settings.test_resamplings.keys()), noizes = [None], N_p = [None], 
                    paths = None, labels_needed = [None], curves = settings.curves, 
                    fission = False, show_n = False, name = '', settings = settings.settings):
    add_str = ''
    if fission:
        add_str+='_fission'
    if show_n:
        add_str+='_N'
    script_dir = get_script_dir() 
    if paths is None:
        densities = [i/settings['size_x']*['size_y'] for i in N_p]
    else:
        densities = [int(a.split('/')[len(a.split('/')) - 1])/settings['size_x']*settings['size_y'] for a in paths]
    test_path = rf"{script_dir}//tests//{N}//"
    x = arange(1, settings['iterations'] + 1)
    errors = []
    if paths is None:
        labels = {'resamplings' : ['' for i in range(len(resamplings))], 'noizes' : ['' for i in range(len(noizes))], 
                    'curves' : ['' for i in range(len(curves))], 'densities' : ['' for i in range(len(densities))]}
        for i, resampling in enumerate(resamplings):
            errors.append([])
            for j, noize in enumerate(noizes):
                errors[i].append([])
                for k, curve in enumerate(curves):
                    errors[i][j].append([])
                    for density in densities:
                        with open(test_path + f'{resampling}//' + f'{noize}//' + f'{curve}//' + f'{round(density)}' + add_str) as f:
                            err_list = [round(float(err.strip()), 5) for err in f]
                        errors[i][j][k].append(err_list)
        
        if 'resamplings' in labels_needed:
            for i, label in enumerate(resamplings):
                labels['resamplings'][i] = 'Алг.:' + label
        if 'noizes' in labels_needed:
            for i, label in enumerate(noizes):
                labels['noizes'][i] = ' Шум:' + str(label) + ' '
        if 'curves' in labels_needed:
            for i, label in enumerate(curves):
                labels['curves'][i] = ' Тр.:' + label
        if 'densities' in labels_needed:
            for i, label in enumerate(densities):
                labels['densities'][i] = ' ПРЧ:' + str(label/(settings.settings['size_x']*settings.settings['size_y']))

        fig, ax = plt.subplots()
        fig.set_figwidth(20)
        fig.set_figheight(23)
        for i in range(len(resamplings)):
            for j in range(len(noizes)):
                for k in range(len(curves)):
                    for l in range(len(densities)):

                        final_label = labels['resamplings'][i] + labels['noizes'][j] + labels['curves'][k] + labels['densities'][l]

                        if i == 0:
                            if densities[l]/100 == 50.1:
                                ax.plot(x, errors[i][j][k][l][0:settings.settings['iterations']], label= final_label, linewidth = 5, linestyle='--')#,marker='X', markersize= 14, )
                            else:
                                ax.plot(x, errors[i][j][k][l][0:settings.settings['iterations']], label= final_label, linewidth = 5, linestyle='--')#,marker='o', markersize= 14, )
                
                        elif i == 1:
                            if densities[l]/100 == 50.1:
                                ax.plot(x, errors[i][j][k][l][0:settings.settings['iterations']], label= final_label, linewidth = 5, linestyle='-.')#,marker='X', markersize= 14, )
                            else:
                                ax.plot(x, errors[i][j][k][l][0:settings.settings['iterations']], label= final_label, linewidth = 5, linestyle='-.')#,marker='o', markersize= 14, )
                                
                        elif i == 2:
                            if densities[l]/100 == 50.1:
                                ax.plot(x, errors[i][j][k][l][0:settings.settings['iterations']], label= final_label, linewidth = 5, linestyle=':')#, marker='X', markersize= 14)
                            else:
                                ax.plot(x, errors[i][j][k][l][0:settings.settings['iterations']], label= final_label, linewidth = 5, linestyle=':')#,marker='o', markersize= 14) 
                        
                        elif i == 3:
                            if densities[l]/100 == 50.1:
                                ax.plot(x, errors[i][j][k][l][0:settings.settings['iterations']], label= final_label, linewidth = 5, linestyle=':')#, marker='X', markersize= 14)
                            else:
                                ax.plot(x, errors[i][j][k][l][0:settings.settings['iterations']], label= final_label, linewidth = 5, linestyle=':')#,marker='o', markersize= 14) 
                        elif i == 4:
                            if densities[l]/100 == 50.1:
                                ax.plot(x, errors[i][j][k][l][0:settings.settings['iterations']], label= final_label, linewidth = 5, linestyle=':')#, marker='X', markersize= 14)
                            else:
                                ax.plot(x, errors[i][j][k][l][0:settings.settings['iterations']], label= final_label, linewidth = 5, linestyle=':')#,marker='o', markersize= 14) 
                        elif i == 5:
                            if densities[l]/100 == 50.1:
                                ax.plot(x, errors[i][j][k][l][0:settings.settings['iterations']], label= final_label, linewidth = 5, linestyle=':', marker='X', markersize= 14)
                            else:
                                ax.plot(x, errors[i][j][k][l][0:settings.settings['iterations']], label= final_label, linewidth = 5, linestyle=':',marker='o', markersize= 14) 
                            
        #ax.yaxis.set_major_locator(ticker.MultipleLocator(0.02))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
        ax.legend(prop={'size': 30}, loc=0)
    else:
        label_final = ''
        fig, ax = plt.subplots()
        fig.set_figwidth(20)
        fig.set_figheight(23)
        for path in paths:

            labels_cur = path.split('/')
            
            if 'resamplings' in labels_needed:
                label_final = label_final + 'Алг.:' + labels_cur[1] + ' '
            if 'noizes' in labels_needed:
                label_final = label_final + 'Шум:' + str(labels_cur[2]) + ' '
            if 'curves' in labels_needed:
                label_final = label_final + 'Скорость:' + labels_cur[3][5] + ' '
            if 'densities' in labels_needed:
                label_final = label_final + 'ПРЧ:' + str(int(labels_cur[4])/100) + ' '

            with open(test_path + path) as f:
                err_list = [round(float(err.strip()), 5) for err in f]

            if labels_cur[1] == 'сист.':
                if labels_cur[3][5] == '1':
                    ax.plot(x, err_list[0:settings.settings['iterations']], label= label_final, linewidth = 5,linestyle='--',marker='X', markersize= 14)
                else:
                    ax.plot(x, err_list[0:settings.settings['iterations']], label= label_final, linewidth = 5,linestyle='--',marker='o', markersize= 14)
                    
            elif labels_cur[1] == 'страт.':
                if labels_cur[3][5] == '1':
                    ax.plot(x,err_list[0:settings.settings['iterations']], label= label_final, linewidth = 5,linestyle='-.',marker='X', markersize= 14)
                else:
                    ax.plot(x, err_list[0:settings.settings['iterations']], label= label_final, linewidth = 5,linestyle='-.',marker='o', markersize= 14)
                    
            elif labels_cur[1] == 'пол.':
                if labels_cur[3][5] == '1':
                    print('e')
                    ax.plot(x,err_list[0:settings.settings['iterations']], label= label_final, linewidth = 5,linestyle=':',marker='X', markersize= 14)
                else:
                    ax.plot(x, err_list[0:settings.settings['iterations']], label= label_final, linewidth =5,linestyle=':',marker='o', markersize= 14)
                            

            label_final = ''
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))

        ax.legend(prop={'size': 30}, loc=1)

    ax.axes.set_aspect(settings.settings['iterations']/plt.ylim()[1]*.8)
    plt.tick_params(axis='both', which='major', labelsize=30)
    plt.xlabel("Итерация", fontsize=45)
    plt.ylabel("ПРЧ", fontsize=45)
    fig.savefig(name, bbox_inches='tight')
    #plt.show()
            
if __name__ == '__main__':
    #visualize_errors(50,paths=['/сист./0.1/speed1/6000', '/пол./0.1/speed1/6000', '/сист./0.1/speed2/6000', '/пол./0.1/speed2/6000'], labels_needed=['resamplings', 'densities', 'curves'], name='speed_comp')
    #visualize_errors(100, noizes=[0.1], curves=['eight'], N_p=[3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000], resamplings=['сист.'], labels_needed=['densities'], name='text_N_comp')
    #N_ccomp_N_50omp_2
    #visualize_errors(100, noizes=[0.2], curves=['eight'], N_p=[1000,2000,3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000], resamplings=['сист.'], labels_needed=['densities'], name='text_N_comp_2')
    
    #visualize_errors(100, noizes=[0.1], curves=['curve'], N_p=[3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000], resamplings=['сист.'], labels_needed=['densities'], name='N_comp')

    #KD_2_comp_res
    #visualize_errors(100, noizes=[0.1], N_p=[6000], name = 'text_KD_2_comp_res', resamplings=['сист. РКЛ 2', 'страт. РКЛ 2', 'пол. РКЛ 2'], curves=['speed1'], labels_needed=['resamplings', 'densities'], fission=False)
    #KD_comp_res
    #visualize_errors(100, noizes=[0.1], N_p=[6000], name = 'text_KD_comp_res', resamplings=['сист. РКЛ', 'страт. РКЛ', 'пол. РКЛ'], curves=['speed1'], labels_needed=['resamplings', 'densities'], fission=False)
    #Pna_comp_res
    #visualize_errors(100, noizes=[0.1], N_p=[6000], name  = 'text_Pna_comp_res', resamplings=['сист. апост.', 'страт. апост.', 'пол. апост.'], curves=['speed1'], labels_needed=['resamplings', 'densities'], fission=False)
    #pres_KD_comp_res
    #visualize_errors(100, noizes=[0.1], N_p=[6000], name = 'pres_KD_comp_res', resamplings=['сист. РКЛ', 'страт. РКЛ', 'пол. РКЛ'], curves=['speed1'], labels_needed=['resamplings', 'densities'], fission=False)
    

    #comp_dens_pna
    #visualize_errors(100, noizes=[0.1], N_p=[5000, 10000], name = 'text_comp_dens_pna', resamplings=['пол. апост.'], curves=['speed1'], labels_needed=['resamplings', 'densities'], fission=False)
    #comp_dens_KD
    #visualize_errors(100, noizes=[0.1], N_p=[5000, 10000], name = 'text_comp_dens_KD', resamplings=['страт. РКЛ'], curves=['speed1'], labels_needed=['resamplings', 'densities'], fission=False)
    #comp_dens_KD_2
    #visualize_errors(100, noizes=[0.1], N_p=[5000, 10000], name = 'text_comp_dens_KD_2', resamplings=['страт. РКЛ 2'], curves=['speed1'], labels_needed=['resamplings', 'densities'], fission=False)
    
    #KD_problems
    #visualize_errors(100, noizes=[0.1], N_p=[5000, 10000], name = 'KD_problems', resamplings=['сист. РК 2', 'сист. РК1'], curves=['speed1'], labels_needed=['resamplings', 'densities'], fission=False)
    #KD_problems_2
    #visualize_errors(100, noizes=[0.1], N_p=[5000], name = 'KD_problems_2_pres', resamplings=['РКЛ стар. пол.', 'РКЛ стар. сист.', 'РКЛ стар. страт.', 'РКЛ нов. пол.', 'РКЛ нов. страт.', 'РКЛ нов. сист.'], curves=['speed1'], labels_needed=['resamplings', 'densities'], fission=False)
    
    #comp_KD
    #visualize_errors(100, noizes=[0.1], N_p=[5000, 10000], name = 'text_comp_KD', resamplings=['пол. РКЛ', 'пол.'], curves=['speed1'], labels_needed=['resamplings', 'densities'], fission=False)
    #comp_KD_2
    #visualize_errors(100, noizes=[0.1], N_p=[5000, 10000], name = 'text_comp_KD_2', resamplings=['пол. РКЛ 2', 'пол.'], curves=['speed1'], labels_needed=['resamplings', 'densities'], fission=False)
    #comp_PNA
    #visualize_errors(100, noizes=[0.1], N_p=[5000, 10000], name = 'text_comp_PNA', resamplings=['пол. апост.', 'пол.'], curves=['speed1'], labels_needed=['resamplings', 'densities'], fission=False)
    
    #comp_all_PNA_100
    #visualize_errors(100, noizes=[0.1], N_p=[10000], name = 'text_comp_all_PNA_100', resamplings=['пол. апост.', 'пол. РКЛ', 'пол. РКЛ 2'], curves=['speed1'], labels_needed=['resamplings', 'densities'], fission=False)
    #comp_all_PNA_50
    #visualize_errors(100, noizes=[0.1], N_p=[5000], name = 'text_comp_all_PNA_50', resamplings=['пол. апост.', 'пол. РКЛ', 'пол. РКЛ 2'], curves=['speed1'], labels_needed=['resamplings', 'densities'], fission=False)
    #visualize_errors(100, noizes=[0.1], N_p=[10000], name = 'comp_all_PNA_50', resamplings=['пол. апост.', 'РКЛ стар. пол.', 'пол. РКЛ 2'], curves=['speed1'], labels_needed=['resamplings', 'densities'], fission=False)

    #comp_N_50
    #visualize_errors(100, noizes=[0.1], N_p=[5000], name = 'text_comp_N_50', resamplings=['пол. апост.', 'пол. РКЛ', 'пол. РКЛ 2'], curves=['speed1'], labels_needed=['resamplings', 'densities'], fission=False, show_n = True)
    #comp_N
    #visualize_errors(100, noizes=[0.1], N_p=[10000], name = 'text_comp_N', resamplings=['пол. апост.', 'пол. РКЛ', 'пол. РКЛ 2'], curves=['speed1'], labels_needed=['resamplings', 'densities'], fission=False, show_n = True)

    #fission_occ
    #visualize_errors(100, noizes=[0.1], N_p=[5000], name = 'fission_occ', resamplings=['пол. РК', 'сист. апост.', 'страт. РК 2'], curves=['speed1'], labels_needed=['resamplings', 'densities'], fission=True, show_n = False)
    #fission_occ_70скорости
    #visualize_errors(100, noizes=[0.1], N_p=[7000], name = 'fission_occ_70', resamplings=['пол. РК', 'сист. апост.', 'страт. РК 2'], curves=['speed1'], labels_needed=['resamplings', 'densities'], fission=True, show_n = False)
    
    #fission_occ_100
    #visualize_errors(100, noizes=[0.1], N_p=[10000], name = 'pres_fission_occ_100', resamplings=['пол. РКЛ', 'сист. апост.', 'страт. РКЛ 2'], curves=['speed1'], labels_needed=['resamplings', 'densities'], fission=True, show_n = False)
    
    #visualize_errors(100, paths=['/пол./0.1/eight/4000', '/пол./0.2/eight/4000', '/сист./0.1/eight/4000', '/сист./0.2/eight/4000', '/страт./0.1/eight/4000', '/страт./0.2/eight/4000'], labels_needed=['noizes', 'resamplings', 'densities'])
    #speed_comp
    #visualize_errors(50, name = 'text_speed_comp',paths=['/пол./0.1/speed1/6000', '/пол./0.1/speed2/6000', '/сист./0.1/speed1/6000', '/сист./0.1/speed2/6000', '/страт./0.1/speed1/6000', '/страт./0.1/speed2/6000'], labels_needed=['resamplings', 'curves'])
    #noizes_comp_80
    #visualize_errors(100, ['пол.', "сист.", "страт."],[0.1, 0.2], [8000],  name='noizes_comp_80', labels_needed=['noizes', 'resamplings'], curves=['eight'])
    #noizes_comp_80_log
    #visualize_errors(100, ['пол.', "сист.", "страт."],[0.1, 0.2], [8000],  name='noizes_comp_80_log', labels_needed=['noizes', 'resamplings'], curves=['eight'])
    #noizes_comp_40
    #visualize_errors(100, ['пол.', "сист.", "страт."],[0.1, 0.2], [4000],  name='noizes_comp_40', labels_needed=['noizes', 'resamplings'], curves=['eight'])
    
    #visualize_errors(100, ["пол."],[0.1], [4000, 5000, 6000, 7000, 8000, 9000, 10000],  name='noizes_comp_40', labels_needed=['noizes', 'resamplings', 'densities'], curves=['eight'])

    #visualize_errors(1, paths=['/mult/0.1/curve/2000', '/mult/0.2/curve/2100'], labels_needed=['noizes', 'resamplings', 'densities'])
    #visualize_errors(1, noizes=[0.1, 0.2], N_p=[2000], curves=['curve'], labels_needed=['noizes', 'resamplings', 'densities', 'curves'])
    
    #visualize_errors(50, name = 'pres_speed_comp',paths=['/пол./0.1/speed1/6000', '/пол./0.1/speed2/6000', '/сист./0.1/speed1/6000', '/сист./0.1/speed2/6000', '/страт./0.1/speed1/6000', '/страт./0.1/speed2/6000'], labels_needed=['resamplings', 'curves'])
    #visualize_errors(100, ['пол.', "сист.", "страт."],[0.1, 0.2], [4000],  name='pres_noizes_comp_40', labels_needed=['noizes', 'resamplings'], curves=['eight'])
    
    pass
