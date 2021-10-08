import numpy as np
from pyrient.orientclasses import Quaternion as Q
from ahrs.filters import Madgwick

def align_to_GCS(acc, gyr, Fs, gain=.1):
    # Compute attitude by fusing measures
    attitude = Madgwick(acc=acc, gyr=gyr, gain=.01, frequency=Fs)

    # Create empty arrays
    a, a_g, a_glob = np.zeros((acc.shape[0], 4)), np.zeros((acc.shape[0], 4)), np.zeros((acc.shape[0]))
    a[:, 1:] = acc

    # Correct for gravity
    for k in range(0, attitude.Q.shape[0]):
        q = Q.Quaternion(attitude.Q[k, :])
        q_conj = q.conj
        a_q = Q.Quaternion(a[k, :])

        a_tmp = q.prod(a_q)
        a_g[k, :] = a_tmp.prod(q_conj).q
        a_glob[k] = a_g[k, 3]

    a_glob -= np.mean(a_glob[0:99])

    return a_glob