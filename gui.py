

from difflib import Match
import dearpygui.dearpygui as dpg
import settings
import main
import tests
from Particle_filter import ParticleFilter
from Robot import model
from copy import deepcopy
import spline

#cur_settings = deepcopy(settings.settings)
#print(cur_settings)

def save_callback(sender, data):
    print("Save Clicked")


def change_settings(sender, app_data, user_data: list[dict, list[str],]):
    for setting in user_data[1]:
        user_data[0].update({setting : app_data})

def change_dict_res(sender, app_data, user_data: list[dict, str,]):
    if app_data:
        user_data[0].update({user_data[1] : user_data[2]})
    else:
        user_data[0].pop(user_data[1])

def change_list(sender, app_data, user_data: list[list]):
    if app_data:
        user_data[0].append(user_data[1])
    else:
        user_data[0].remove(user_data[1])

def change_list_res(sender, app_data, user_data: list[list]):
    if app_data:
        user_data[0]["mode"] = "change_n"
        user_data[1].append(user_data[2])
    else:
        user_data[0]["mode"] = "fast"
        user_data[1].remove(user_data[2])


def model_callback(sender, app_data, user_data):
    if app_data:
        with dpg.collapsing_header(label="Model settings", tag="model_settings", parent="change_tr"):
                dpg.add_input_float(label="Probability of left rotation", tag="pr_left", default_value= 0.4, max_value= 1)
                dpg.add_input_float(label="Probability of right rotation", tag="pr_right", default_value= 0.4, max_value= 1)
                dpg.add_input_float(label="Max angle of rotation", tag="max_ang", default_value=180, max_value=360)
                dpg.add_input_float(label="Speed of robot", tag="speed")
        user_data = None
    else:
        dpg.delete_item("model_settings")

def gen_curve(sender, app_data, user_data: list):
    if app_data:
        dpg.add_input_int(label="Number of points for B-spline", tag="N_p_spline", parent="change_tr", default_value=10, min_value=2)
        for i in range(dpg.get_value("num_i")):
            points = spline.generate_points(dpg.get_value("N_p_spline"))
            curve = spline.create_curve(points)
            user_data.insert(0, curve)
    else:
        dpg.delete_item("N_p_spline")
        for i in range(min(len(user_data), dpg.get_value("num_i"))):
            user_data.pop(0)

def run_main(sender, app_data, user_data):
    errors = []
    errors.append(tests.test_curves(dpg.get_value("N_p"), 
                                    dpg.get_value("N_p")+1, 
                                    1, 
                                    1, 
                                    dpg.get_value("noize"),
                                    dpg.get_value("noize")+1,
                                    1,
                                    None if not dpg.get_value("use_model") 
                                    else model(
                                        dpg.get_value("pr_left"),
                                        dpg.get_value("pr_right"),
                                        dpg.get_value("max_ang"),
                                        dpg.get_value("speed")
                                    ),
                                    user_data[0],
                                    user_data[1],
                                    user_data[2],
                                    user_data[3]))


def run_tests(sender, app_data, user_data):
    tests.test_curves(dpg.get_value("L_N_p"), 
                      dpg.get_value("H_N_p"), 
                      dpg.get_value("step_N_p"), 
                      dpg.get_value("tests_N"), 
                      dpg.get_value("noize_L"),
                      dpg.get_value("noize_H"),
                      dpg.get_value("noize_step"),
                      None if not dpg.get_value("use_model") 
                      else model(
                          dpg.get_value("pr_left"),
                          dpg.get_value("pr_right"),
                          dpg.get_value("max_ang"),
                          dpg.get_value("speed")
                      ),
                        user_data[0],
                        user_data[1],
                        user_data[2],
                        user_data[3])
                        

def callback_single_multiple(sender, app_data, user_data: list[dict, dict, list, list]):

    if app_data == "Single test":
        user_data[1].update({"mult": ParticleFilter.multinomial_resample})
        user_data[2].append(_)
        user_data[3].append(spline.get_curves()[1])

        dpg.delete_item("change_res")
        dpg.delete_item("change_N_res")
        dpg.delete_item("change_tr")
        dpg.delete_item("change_tests")
        dpg.delete_item("run_mult")


        dpg.add_input_int(label="Number of particles", tag="N_p", callback=change_settings, 
                        user_data=(user_data[0], ["N_p"]), default_value=5000, before="num_i")
        dpg.add_input_float(label="Noize", tag="noize", callback=change_settings, 
                        user_data=(user_data[0], ["noize"]), default_value=0.1, before="num_i", parent="test_pf")
        
        with dpg.collapsing_header(label="Usual resamplings", tag="change_res", parent="test_pf"):
            dpg.add_radio_button(["Multinomial resampling", "Systematic resampling", "Stratified resampling"], tag="radio_res", parent="change_res")
        with dpg.collapsing_header(label="Change N_p resamplings", tag="change_N_res", parent="test_pf"):
            dpg.add_radio_button(["KD resampling", "KD 2 rows resampling", "Based on posterior resampling"], tag="radio_N_res", parent="change_N_res")
        with dpg.collapsing_header(label="Trajectory", tag="change_tr", parent="test_pf"):
            dpg.add_radio_button(["model", "Eight", "Previously generated", "Generated"], tag="radio_tr", parent="change_tr")
        
        dpg.add_button(label="Run", tag="run_sing", callback=run_main, user_data=(user_data[0], user_data[1], user_data[2], user_data[3]), parent="test_pf")


    elif app_data == "Multiple tests":

        dpg.delete_item("change_res")
        dpg.delete_item("change_N_res")
        dpg.delete_item("change_tr")
        dpg.delete_item("N_p")
        dpg.delete_item("noize")
        dpg.delete_item("run_sing")

        with dpg.collapsing_header(label="Usual resamplings", tag="change_res", parent="test_pf"):
            dpg.add_checkbox(label="Multinomial resampling", callback= change_dict_res, 
                            user_data=(user_data[1], "mult", ParticleFilter.multinomial_resample), parent="change_res")
            dpg.add_checkbox(label="Systematic resampling", callback= change_dict_res, 
                            user_data=(user_data[1], "syst", ParticleFilter.systematic_resample), parent="change_res")
            dpg.add_checkbox(label="Stratified resampling", callback= change_dict_res, 
                            user_data=(user_data[1], "mult", ParticleFilter.stratified_resample), parent="change_res")

        with dpg.collapsing_header(label="Change N_p resamplings", tag="change_N_res", parent="test_pf"):
            dpg.add_checkbox(label="KD resampling", callback= change_list_res, user_data=(user_data[0], user_data[2], "KD"), parent="change_N_res")
            dpg.add_checkbox(label="KD 2 rows resampling", callback= change_list_res, user_data=(user_data[0], user_data[2], "KD_2"), parent="change_N_res")
            dpg.add_checkbox(label="Based on posterior resampling", callback= change_list_res, user_data=(user_data[0], user_data[2], "Pna"), parent="change_N_res")
            
        with dpg.collapsing_header(label="Trajectory", tag="change_tr", parent="test_pf"):
            dpg.add_checkbox(label="model", tag="use_model", callback=model_callback, user_data=user_data[3], parent="change_tr")
            dpg.add_checkbox(label="Eight", tag="eight", callback= change_list, user_data=(user_data[3], spline.get_curves()[1]), parent="change_tr")
            dpg.add_checkbox(label="Previously generated", tag="pr_gen", callback= change_list, user_data=(user_data[3], spline.get_curves()[0]), parent="change_tr")
            dpg.add_checkbox(label="Generated", tag="gen", callback= gen_curve, user_data=user_data[3], parent="change_tr")
        
        with dpg.collapsing_header(label="Tests settings", tag="change_tests", parent="test_pf"):
            dpg.add_input_int(label="Lowest number of particles", tag="L_N_p", default_value= 1000, min_value= 1)
            dpg.add_input_int(label="Highest number of particles", tag="H_N_p", default_value= 5000, min_value= 1)
            dpg.add_input_int(label="Step of number of particles", tag="step_N_p", default_value= 500, min_value= 1)
            dpg.add_input_float(label="Lowest value of noizes", tag="noize_L", default_value= 0.1, min_value= 0)
            dpg.add_input_float(label="Highest value of noizes", tag="noize_H", default_value= 0.2, min_value= 0)
            dpg.add_input_float(label="Step of value of noizes", tag="noize_step", default_value= 0.05, min_value= 0)
            dpg.add_input_int(label="Number of tests", tag="tests_N", default_value=10, min_value= 1)

        dpg.add_button(label="Run Tests", tag="run_mult", callback=run_tests, user_data=(user_data[0], user_data[1], user_data[2], user_data[3]), parent="test_pf")
        
            

def radio_res(sender, app_data, user_data: dict):
    match app_data:
        case "Multinomial resampling":
            user_data.pop()
            user_data.update({"mult": ParticleFilter.multinomial_resample})

        case "Systematic resampling":
            user_data.pop()
            user_data.update({"syst": ParticleFilter.systematic_resample})

        case "Stratified resampling":
            user_data.pop()
            user_data.update({"strat": ParticleFilter.stratified_resample})
        
def radio_N_res(sender, app_data, user_data: list[dict, list]):
    user_data[0]["mode"] = "change_n"
    match app_data:
        case "KD resampling":
            user_data[1].append("KD")
        case "KD 2 rows resampling":
            user_data[1].append("KD_2")
        case "Based on posterior resampling":
            user_data[1].append("Pna")

def radio_tr(sender, app_data, user_data: list):
    match app_data:
        case "Model":
            model_callback(user_data=user_data[1])
        case "Eight":

        case "Previously generated":

        case "Generated":

def radio_change_N(sender, app_data):
    if app_data:
         with dpg.collapsing_header(label="Change N_p resamplings", tag="change_N_res"):
                dpg.add_radio_button(["KD resampling", "KD 2 rows resampling", "Based on posterior resampling"], tag="radio_N_res", 
                                        callback=radio_N_res, user_data= (cur_settings, change_n), parent="test_pf", before="change_tr")
    else:
        dpg.delete_item("change_N_res")


        

dpg.create_context()
dpg.create_viewport(title= "Particle Filter", width=600, height= 800)
dpg.setup_dearpygui()

with dpg.window(label="Settings", tag="Primary Window"):
    
    cur_settings = deepcopy(settings.settings)
    test_resamplings = {}
    change_n = []
    curves = []
    with dpg.tab_bar():
        with dpg.tab(label="Test particle filter", tag="test_pf"):
        
            dpg.add_radio_button(["Single test", "Multiple tests"], callback=callback_single_multiple, 
                                user_data= (cur_settings, test_resamplings, change_n, curves), horizontal=True)
            dpg.add_input_int(label="Number of particles", tag="N_p", callback=change_settings, 
                                user_data=(cur_settings, ["N_p"]), default_value=5000, before="num_i")
            dpg.add_input_float(label="Noize", tag="noize", callback=change_settings, 
                                user_data=(cur_settings, ["noize_dist", "noize_rot", "noize_sens"]), default_value=0.1, before="num_i")
        
            dpg.add_input_int(label="Number of iterations", tag="num_i", callback=change_settings, 
                                user_data= (cur_settings, ["iterations"]), default_value=12)
            dpg.add_input_int(label="Field size X", callback=change_settings, user_data= (cur_settings, ["size_x"]), default_value=10)
            dpg.add_input_int(label="Field size Y", callback=change_settings, user_data= (cur_settings, ["size_y"]), default_value=10)
            
            dpg.add_checkbox(label="Use change N apgotythms", tag="radio_change_N", callback=radio_change_N)
            with dpg.collapsing_header(label="Usual resamplings", tag="change_res"):
                dpg.add_radio_button(["Multinomial resampling", "Systematic resampling", "Stratified resampling"], tag="radio_res", 
                                        callback=radio_res, user_data= test_resamplings)
            with dpg.collapsing_header(label="Trajectory", tag="change_tr"):
                dpg.add_radio_button(["Model", "Eight", "Previously generated", "Generated"], tag="radio_tr", callback=radio_tr, user_data=curves)

            dpg.add_button(label="Run", tag="run_sing", callback=run_main, user_data=(cur_settings, test_resamplings, change_n, curves))
        with dpg.tab(label="Check errors"):
            pass
dpg.show_viewport()
dpg.set_primary_window("Primary Window", True)
dpg.start_dearpygui()
dpg.destroy_context()