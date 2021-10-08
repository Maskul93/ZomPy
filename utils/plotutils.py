import numpy as np
import matplotlib.pyplot as plt
# from common import Jump

def plot_raw(acc, gyr, Fs):
    fig, axs = plt.subplots(2)
    fig.suptitle('Raw Accelerometer and Gyroscope Data')
    fig.dpi = 70
    t = np.linspace(0, acc.shape[0] / Fs, num=acc.shape[0])
    axs[0].plot(t, acc, linewidth=1)
    axs[0].set_xlim(0, 5)
    axs[0].set_ylabel('Acceleration (m/s$^2$)')
    axs[1].plot(t, gyr, linewidth=1)
    axs[1].set_xlim(0, 5)
    axs[1].set_ylabel('Angular Velocity (rad/s)')
    plt.xlabel("Time (s)")

    return fig

def plot_aligned(a_glob, Fs, acc_v):
    fig = plt.figure()
    fig.dpi = 70
    t = np.linspace(0, a_glob.shape[0] / Fs, num=a_glob.shape[0])
    plt.plot(t, a_glob, linewidth=1)
    plt.plot(t, acc_v, linewidth=1)
    plt.xlim([0, 5])
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s$^2$)')
    fig.suptitle('Corrected and Original Acceleration')
    plt.legend(["Corrected", "Raw w/o g"])

    return fig

def plot_features(jump, Fs):
    fig1 = jump.plot(dpi=70)
    fig1.suptitle('Time Duration of each CMJ phase')
    fig2, fig3 = jump.features.plot(jump, dpi=70)
    fig2.suptitle('First set of features')
    fig3.suptitle('Second set of features')
    fig4 = plt.figure(dpi=70)

    t = np.linspace(0, jump.features.u1.shape[0] / Fs, num=jump.features.u1.shape[0])
    plt.plot(t, jump.features.u3, linewidth=1)
    plt.plot(t, jump.features.u2, linewidth=1)
    plt.plot(t, jump.features.u1, linewidth=1)
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s$^2$)')
    plt.xlim(0, 5)
    fig4.suptitle('IMFs computed through VMD')
    plt.legend(['$\omega_3$', '$\omega_2$', '$\omega_1$'])

    return fig1, fig2, fig3, fig4

def plot_velocity(jump):
    v = jump.v[jump.t_0:jump.t_TO]
    t = np.linspace(0, v.shape[0] / jump.fs, num=v.shape[0])
    fig = plt.figure(dpi=100)
    fig.suptitle('Velocity from $t_0$ to $t_{TO}$')
    plt.plot(t, v)
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.xlim(0, t[-1])

    return fig