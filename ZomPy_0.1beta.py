import PySimpleGUI as sg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Note the matplot tk canvas import
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from ahrs.filters import Madgwick
from pyrient.orientclasses import Quaternion as Q
from common import Jump
from utils import plotutils as pu
from utils import margutils as mu


# VARS CONSTS:
_VARS = {'window': False}

def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

# \\  -------- Main GUI -------- //
AppFont = 'Any 16'
sg.theme('LightBlue')

layout = [
            [sg.Canvas(key='-raw_data-'), sg.Canvas(key='-aligned_data-'), sg.Canvas(key='-IMFs-')],
            [sg.Canvas(key='-features_1-'), sg.Canvas(key='-features_2-'), sg.Canvas(key='-features_3-')],
            [sg.Input(key='-INPUT-'), sg.FileBrowse(), sg.Button('Analyze')],
            [sg.Text('Fs', size=(2, 1)), sg.InputText(key='-sampl_freq-', size=(4, 1), default_text="128"),
            sg.Text('β', size=(1, 1)), sg.InputText(key='-mad_gain-', size=(4, 1), default_text="0.01"),
            sg.InputCombo(('IMU', 'Forceplate'), size=(11, 1), default_value='IMU')],
            [sg.Button('Plot Raw'), sg.Button('Plot Aligned'), sg.Button('Visualize Features'), sg.Button('Velocity')],
            [sg.Button('Exit')]
            ]

_VARS['window'] = sg.Window('ZomPy 0.1 beta',
                            layout,
                            finalize=False,
                            resizable=True,
                            element_justification='left')

# Window 2 layout
layout2 = [[sg.Canvas(key='-vel_data-')], [[sg.Output(size=(60, 10))]],
           [sg.Button('Close')]]
# Window 2 initialization
_VARS['window2'] = sg.Window('Velocity and Height',
                             layout2)

# MAIN LOOP
while True:
    event, values = _VARS['window'].read(timeout=200)
    if event == sg.WIN_CLOSED or event == 'Exit':
        break
    
    if event == 'Analyze':
        loaded_file = values['-INPUT-']     # File path
        get_fs = values['-sampl_freq-']     # Sampling frequency (string)
        get_beta = values['-mad_gain-']     # Madgwick filter gain (string)

        Fs = int(get_fs)        # Convert the user input into usable integer
        gain = float(get_beta)  # Convert the user input into usable float

        file_IMU = pd.read_csv(loaded_file)         # Load file
        acc = pd.DataFrame.to_numpy(file_IMU[["ax", "ay", "az"]])       # Accelerometer measures
        gyr = pd.DataFrame.to_numpy(file_IMU[["gx", "gy", "gz"]])       # Gyroscope measures
        t = np.linspace(0, acc.shape[0] / Fs, num=acc.shape[0])       # Time array
        a_glob = mu.align_to_GCS(acc=acc, gyr=gyr, Fs=Fs, gain=gain)       # Correct for trunk bending
        acc_v = acc[:, 1] - np.mean(acc[0:99,1])         # Obtain a comparable acceleration trace for '-plot_aligned-'
        jump = Jump.CMJ(a_glob, fs=Fs)        # Create the 'Jump' object containing the jump aligned with the GCS

    if event == 'Plot Raw':
        fig = pu.plot_raw(acc=acc, gyr=gyr, Fs=Fs)
        draw_figure(_VARS['window']['-raw_data-'].TKCanvas, fig)

    if event == 'Plot Aligned':
        fig = pu.plot_aligned(a_glob=a_glob, Fs=Fs, acc_v=acc_v)
        draw_figure(_VARS['window']['-aligned_data-'].TKCanvas, fig)

    if event == 'Visualize Features':
        fig1, fig2, fig3, fig4 = pu.plot_features(jump=jump, Fs=Fs)
        draw_figure(_VARS['window']['-features_1-'].TKCanvas, fig1)
        draw_figure(_VARS['window']['-features_2-'].TKCanvas, fig2)
        draw_figure(_VARS['window']['-features_3-'].TKCanvas, fig3)
        draw_figure(_VARS['window']['-IMFs-'].TKCanvas, fig4)

    if event == 'Velocity':
        event2, values2 = _VARS['window2'].read(timeout=100)
        if event2 == sg.WIN_CLOSED or event2 == 'Close':
            # Close this fucking window dc
            #break

        fig = pu.plot_velocity(jump)
        draw_figure(_VARS['window2']['-vel_data-'].TKCanvas, fig)
        s1 = 'Velocity at Take-Off: ' + str(round(jump.v[jump.t_TO],2)) + 'm/s\n'
        s2 = 'Jump Height: ' + str(round(jump.h, 2)) + 'm'
        print(s1+s2)

_VARS['window'].close()