from email.policy import default
from os import path
from bokeh.plotting import figure, output_file, save
from bokeh.io import show
from bokeh.models import ColumnDataSource, CustomJS, Slider
from bokeh.layouts import column
from matplotlib.pyplot import legend
import settings
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tools import get_script_dir
from numpy import arange, asarray

def visualize_filter(Robot_hist, particles_hist, mean_hist, size_x, size_y, N_p, iterations, error_hist, curve = None):
    Robot_data = {}
    particles_data = {}
    mean_data = {}
    error_hist = asarray(error_hist)[:, 0]
    error_data = {}
    for i, data in enumerate(Robot_hist):
        Robot_data.update({i : {'X' : [data[0][0]], 'Y' : [data[0][1]], 'Rotation' : [data[1]]}})
    for i, data in enumerate(particles_hist):
        particles_data.update({i : {'X' : data[:,0], 'Y' : data[:,1], 'Rotation' : data[:,2]}})
    for i, data in enumerate(mean_hist):
        mean_data.update({i : {'X' : [data[0]], 'Y' : [data[1]], 'Rotation' : [data[2]]}})
    for i, data in enumerate(error_hist):
        error_data.update({i : {'Error' : [f'Error: {round(data, 5)}']}})

    pl = figure(plot_width = 800, plot_height = 800, 
                    x_range = (0, size_x), y_range = (0, size_y), 
                    title = 'Particle Filter'
        )
    source_visible_particles = ColumnDataSource(particles_data[0])
    source_visible_robot = ColumnDataSource(Robot_data[0])
    source_visible_mean = ColumnDataSource(mean_data[0])
    source_visible_error = ColumnDataSource(error_data[0])

    pl.triangle(
        'X', 'Y', angle = 'Rotation',
        size = 10, 
        fill_color = "violet",
        line_color = "violet",
        fill_alpha = 0.10,
        line_width = 1,
        legend_label = f"Particles density\nN = {int(N_p/100)}",
        source = source_visible_particles
      )
    pl.triangle_dot(
        'X', 'Y', angle = 'Rotation',
        size = 18,
        fill_color = 'green',
        line_color = 'green',
        fill_alpha = 0.15,
        legend_label = "Robot",
        source = source_visible_robot
    )
    pl.triangle_dot(
        'X', 'Y', angle = 'Rotation',
        size = 18,
        fill_color = 'red',
        line_color = 'red',
        fill_alpha = 0.15,
        legend_label = "Mean",
        source = source_visible_mean
    )
    pl.circle(
        source = source_visible_error,
        x = 0, y = 0, size=0, 
        legend='Error'
    )
    
    if curve is not None:
        pl.line(x = curve[:,0], y = curve[:,1], line_width = 3)
    else:
        xs = list(map(lambda item: item['X'], Robot_data.values()))
        ys = list(map(lambda item: item['Y'], Robot_data.values()))
        pl.line(x = xs, y = ys, line_width = 3)

    if not settings.settings['detail']:
        slider = Slider(start=0, end=iterations, value=0, step=1, title="Iteration")
    else:
        slider = Slider(start=0, end=2*iterations, value=0, step=1, title="Stage")
    callback = CustomJS(args=dict(source_particles=particles_data, 
                                  source_robot = Robot_data,
                                  source_mean = mean_data,
                                  source_error = error_data,
                                  source_visible_particles = source_visible_particles,
                                  source_visible_robot = source_visible_robot,
                                  source_visible_mean = source_visible_mean,
                                  source_visible_error = source_visible_error,
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
    """)
    pl.legend.label_text_font_size = '16pt'
    slider.js_on_change('value', callback)
    
    layout = column(slider, pl)
    #save(layout)
    show(layout)

def visualize_errors(N, resamplings = settings.resamplings, noizes = [None], N_p = [None], 
                    paths = None, labels_needed = [None], curves = settings.curves):
    print(resamplings)
    #root = tk.Tk()
    script_dir = get_script_dir() 
    if paths is None:
        densities = [i/settings.settings['size_x']*settings.settings['size_y'] for i in N_p]
    else:
        densities = [int(a.split('/')[len(a.split('/')) - 1])/settings.settings['size_x']*settings.settings['size_y'] for a in paths]
    test_path = rf"{script_dir}//tests//{N}//"
    x = arange(1, settings.settings['iterations'] + 1)
    #labels = {'resampling' : {}.fromkeys(resamplings), 'noizes' : {}.fromkeys(noizes), 'curves' : {}.fromkeys(curves), 'resa' : {}.fromkeys()}.fromkeys(densities)
    #labels = {['' for i in range(len(resamplings))], ['' for i in range(len(noizes))], ['' for i in range(len(curves))], ['' for i in range(len(densities))]]
    errors = []
    if paths is None:
        #labels = {'resamplings' : [], 'noizes' : [], 'curves' : [], 'densities' : []}
        labels = {'resamplings' : ['' for i in range(len(resamplings))], 'noizes' : ['' for i in range(len(noizes))], 
                    'curves' : ['' for i in range(len(curves))], 'densities' : ['' for i in range(len(densities))]}
        for i, resampling in enumerate(resamplings):
            errors.append([])
            for j, noize in enumerate(noizes):
                errors[i].append([])
                for k, curve in enumerate(curves):
                    errors[i][j].append([])
                    for density in densities:
                        with open(test_path + f'{resampling}//' + f'{noize}//' + f'{curve}//' + f'{round(density)}') as f:
                            err_list = [round(float(err.strip()), 5) for err in f]
                        errors[i][j][k].append(err_list)
        
        
        if 'resamplings' in labels_needed:
            for i, label in enumerate(resamplings):
                labels['resamplings'][i] = 'Res:' + label
        if 'noizes' in labels_needed:
            for i, label in enumerate(noizes):
                labels['noizes'][i] = ' Noize:' + label.__str__()
        if 'curves' in labels_needed:
            for i, label in enumerate(curves):
                labels['curves'][i] = ' Tr:' + label
        if 'densities' in labels_needed:
            for i, label in enumerate(densities):
                labels['densities'][i] = ' Density:' + label.__str__()
        print(labels)
        fig, ax = plt.subplots()
        for i in range(len(resamplings)):
            for j in range(len(noizes)):
                for k in range(len(curves)):
                    for l in range(len(densities)):
                        final_label = labels['resamplings'][i] + labels['noizes'][j] + labels['curves'][k] + labels['densities'][l]
                        ax.plot(x, errors[i][j][k][l], label= final_label)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.legend(prop={'size': 20})
    else:
        label_final = ''
        fig, ax = plt.subplots()
        for path in paths:

            labels_cur = path.split('/')
            print(labels_cur)
            if 'resamplings' in labels_needed:
                label_final = label_final + 'Res.:' + labels_cur[1] + ' '
            if 'noizes' in labels_needed:
                label_final = label_final + 'Noize:' + labels_cur[2].__str__() + ' '
            if 'curves' in labels_needed:
                label_final = label_final + 'Tr:' + labels_cur[3] + ' '
            if 'densities' in labels_needed:
                label_final = label_final + 'Density:' + labels_cur[4].__str__() + ' '

            with open(test_path + path) as f:
                err_list = [round(float(err.strip()), 5) for err in f]
            #print(x, err_list, label_final)
            ax.plot(x, err_list, label= label_final)
            label_final = ''
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.legend(prop={'size': 20})
    fig.set_figwidth(20)
    fig.set_figheight(15)
    #root.mainloop()
    plt.show()
            
if __name__ == '__main__':
    #visualize_errors(1, paths=['/mult/0.1/curve/2000', '/mult/0.2/curve/2100'], labels_needed=['noizes', 'resamplings', 'densities'])
    visualize_errors(1, noizes=[0.1, 0.2], N_p=[2000], curves=['curve'], labels_needed=['noizes', 'resamplings', 'densities', 'curves'])

