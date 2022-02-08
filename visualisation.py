from bokeh.plotting import figure, output_file, save
from bokeh.io import show
from bokeh.models import ColumnDataSource, CustomJS, Slider
from bokeh.layouts import column
from settings import *

def visualize(Robot_hist, particles_hist, mean_hist, curve = None):

    pl = figure(plot_width = 800, plot_height = 800, 
                    x_range = (0, size_x), y_range = (0, size_y), 
                    title = 'Particle Filter'
        )
    source_visible_particles = ColumnDataSource(particles_hist[0])
    source_visible_robot = ColumnDataSource(Robot_hist[0])
    source_visible_mean = ColumnDataSource(mean_hist[0])
    
    pl.triangle(
        'X', 'Y', angle = 'Rotation',
        size = 10, 
        fill_color = "violet",
        line_color = "violet",
        fill_alpha = 0.10,
        line_width = 1,
        legend_label = f"Particles\nN = {N_p}",
        source = source_visible_particles
      )
    pl.triangle_dot(
        'X', 'Y', angle = 'Rotation',
        size = 15,
        fill_color = 'yellow',
        line_color = 'yellow',
        fill_alpha = 0.15,
        legend_label = "Robot",
        source = source_visible_robot
    )
    pl.triangle_dot(
        'X', 'Y', angle = 'Rotation',
        size = 15,
        fill_color = 'red',
        line_color = 'red',
        fill_alpha = 0.15,
        legend_label = "Mean",
        source = source_visible_mean
    )
    xs = list(map(lambda item: item['X'], Robot_hist.values()))
    ys = list(map(lambda item: item['Y'], Robot_hist.values()))
    print(type(curve))
    if curve is not None:
        pl.line(x = curve[:,0], y = curve[:,1])
    else:
        pl.line(x = xs, y = ys)
    print(Robot_hist, '\n\n\n\n', mean_hist)
    Iter_slider = Slider(start=0, end=iterations, value=0, step=1, title="Iteration")
    callback = CustomJS(args=dict(source_particles=particles_hist, 
                                  source_robot = Robot_hist,
                                  source_mean = mean_hist,
                                  source_visible_particles = source_visible_particles,
                                  source_visible_robot = source_visible_robot,
                                  source_visible_mean = source_visible_mean,
                                  iteration=Iter_slider), 
                                    code="""
    var iter = iteration.value;
    var data_visible_particles = source_visible_particles.data;
    var data_available_particles = source_particles.data;
    data_visible_particles.X = source_particles[iter].X
    data_visible_particles.Y = source_particles[iter].Y
    data_visible_particles.Rotation = source_particles[iter].Rotation
    source_visible_particles.change.emit();
    
    var data_visible_robot = source_visible_robot.data;
    var data_available_robot = source_robot.data;
    data_visible_robot.X = source_robot[iter].X
    data_visible_robot.Y = source_robot[iter].Y
    data_visible_robot.Rotation = source_robot[iter].Rotation
    source_visible_robot.change.emit();
    
    var data_visible_mean = source_visible_mean.data;
    var data_available_mean = source_mean.data;
    data_visible_mean.X = source_mean[iter].X
    data_visible_mean.Y = source_mean[iter].Y
    data_visible_mean.Rotation = source_mean[iter].Rotation
    source_visible_mean.change.emit();
    """)
    #print(particles_hist)
    Iter_slider.js_on_change('value', callback)
    
    layout = column(Iter_slider, pl)
    save(layout)
    show(layout)