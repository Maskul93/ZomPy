import PySimpleGUI as sg
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Note the matplot tk canvas import
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from ahrs.filters import Madgwick
from pyrient.orientclasses import Quaternion as Q
import Jump


# VARS CONSTS:
_VARS = {'window': False}

def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

def align_to_GCS(acc, gyr, Fs, gain = .1):
    # Compute attitude by fusing measures
    attitude = Madgwick(acc = acc, gyr = gyr, gain = .01, frequency = Fs)

    # Create empty arrays
    a, a_g, a_glob = np.zeros((acc.shape[0],4)), np.zeros((acc.shape[0],4)), np.zeros((acc.shape[0]))
    a[ : , 1 :] = acc

    # Correct for gravity
    for k in range(0, attitude.Q.shape[0]):
        q = Q.Quaternion(attitude.Q[k , :])
        q_conj = q.conj
        a_q = Q.Quaternion(a[k, :])   

        a_tmp = q.prod(a_q)
        a_g[k, :] = a_tmp.prod(q_conj).q
        a_glob[k] = a_g[k,3]
    
    a_glob -= np.mean(a_glob[0:99])

    return a_glob

# \\  -------- Main GUI -------- //
AppFont = 'Any 16'
sg.theme('LightBlue')

layout = [
            [sg.Canvas(key = '-raw_data-'), sg.Canvas(key = '-aligned_data-'), sg.Canvas(key = '-IMFs-')],
            [sg.Canvas(key = '-features_1-'), sg.Canvas(key = '-features_2-'), sg.Canvas(key = '-features_3-')],
            [sg.Input(key = '-INPUT-'), sg.FileBrowse(), sg.Button('Analyze')],
            [sg.Text('Fs', size =(2, 1)), sg.InputText(key = '-sampl_freq-', size = (4,1), default_text = "128"),
            sg.Text('β', size = (1, 1)), sg.InputText(key = '-mad_gain-', size = (4,1), default_text = "0.01"),
            sg.InputCombo(('IMU', 'Forceplate'), size=(11, 1), default_value = 'IMU')],      
            [sg.Button('Plot Raw'), sg.Button('Plot Aligned'), sg.Button('Visualize Features')],
            [sg.Button('Exit', font  = AppFont)]
            ]

_VARS['window'] = sg.Window('ZomPy 0.1 beta',
                            layout,
                            finalize = False,
                            resizable = True,
                            element_justification = 'left')



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
        t = np.linspace(0, acc.shape[0] / Fs, num = acc.shape[0])       # Time array
        a_glob = align_to_GCS(acc = acc, gyr = gyr, Fs = Fs, gain = gain)       # Correct for trunk bending
        acc_v = acc[:,1] - np.mean(acc[0:99,1])         # Obtain a comparable acceleration trace for '-plot_aligned-'
        jump = Jump.CMJ(a_glob, fs = Fs)        # Create the 'Jump' object containing the jump aligned with the GCS

    if event == "Plot Raw":
        fig, axs = plt.subplots(2)
        fig.dpi = 70
        fig.suptitle('Raw Accelerometer and Gyroscope Data')
        axs[0].plot(t, acc, linewidth = 1)
        axs[0].set_xlim(0, 5)
        axs[0].set_ylabel('Acceleration (m/s$^2$)')
        axs[1].plot(t, gyr, linewidth = 1)
        axs[1].set_xlim(0, 5)
        axs[1].set_ylabel('Angular Velocity (rad/s)')
        plt.xlabel("Time (s)")

        draw_figure(_VARS['window']['-raw_data-'].TKCanvas, fig)

    if event == 'Plot Aligned':
        fig = plt.figure()
        fig.dpi = 70
        t = np.linspace(0, a_glob.shape[0] / Fs, num = a_glob.shape[0])
        plt.plot(t, a_glob, linewidth = 1)
        plt.plot(t, acc_v, linewidth = 1)
        plt.xlim([0, 5])
        plt.xlabel('Time (s)')
        plt.ylabel('Acceleration (m/s$^2$)')
        fig.suptitle('Corrected and Original Acceleration')
        plt.legend(["Corrected", "Raw w/o g"])
        draw_figure(_VARS['window']['-aligned_data-'].TKCanvas, fig)

    if event == 'Visualize Features':
        fig1 = jump.plot(dpi = 70)
        fig1.suptitle('Time Duration of each CMJ phase')
        draw_figure(_VARS['window']['-features_1-'].TKCanvas, fig1)
        
        fig2, fig3 = jump.features.plot(jump, dpi = 70)
        fig2.suptitle('First set of features')
        fig3.suptitle('Second set of features')
        draw_figure(_VARS['window']['-features_2-'].TKCanvas, fig2)
        draw_figure(_VARS['window']['-features_3-'].TKCanvas, fig3)

        fig4 = plt.figure()
        fig4.dpi = 70
        t = np.linspace(0, jump.features.u1.shape[0] / Fs, num = jump.features.u1.shape[0])
        plt.plot(t, jump.features.u3, linewidth = 1)
        plt.plot(t, jump.features.u2, linewidth = 1)
        plt.plot(t, jump.features.u1, linewidth = 1)
        plt.xlabel('Time (s)')
        plt.ylabel('Acceleration (m/s$^2$)')
        plt.xlim(0,5)
        fig4.suptitle('IMFs computed through VMD')
        plt.legend(['$\omega_3$', '$\omega_2$', '$\omega_1$'])
        draw_figure(_VARS['window']['-IMFs-'].TKCanvas, fig4)

_VARS['window'].close()