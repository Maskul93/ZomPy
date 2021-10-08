"""
MADGWICK AHRS FILTER
====================
"""
import numpy as np
from numpy.linalg import norm 
from ..orientclasses.Quaternion import Quaternion 

class MadgwickAHRS:
    def __init__(self, fs = 100, q0 = Quaternion(np.array([1, 0, 0, 0])), beta = .1):
        self.fs = fs
        self.ts = 1 / fs
        self.q0 = q0
        self.beta = beta

    ## --- For Magnetic and Inertial Sensors --- ##
    def UpdateMIMU(self, acc, gyr, mag):
        
        q = self.q0
        
        # Make sure that each measure is squeezed to 1D -- (3,)
        acc = np.array(acc, dtype = float).flatten()
        gyr = np.array(gyr, dtype = float).flatten()
        mag = np.array(mag, dtype = float).flatten()

        # Normalize accelerometer and magnetometer measures
        acc /= norm(acc)
        mag /= norm(mag)

        qmag = Quaternion(np.array([0., mag[0], mag[1], mag[2]]))
        h = q.prod( qmag.prod( q.conj ))
        b = np.array([0., norm(h.q[1:3]), 0., h.q[3]])

        # Apply gradient descent

        f = np.array([
            2*(q.q[1]*q.q[3] - q.q[0]*q.q[2]) - acc[0],
            2*(q.q[0]*q.q[1] + q.q[2]*q.q[3]) - acc[1],
            2*(0.5 - q.q[1]**2 - q.q[2]**2) - acc[2],
            2*b[1]*(0.5 - q.q[2]**2 - q.q[3]**2) + 2*b[3]*(q.q[1]*q.q[3] - q.q[0]*q.q[2]) - mag[0],
            2*b[1]*(q.q[1]*q.q[2] - q.q[0]*q.q[3]) + 2*b[3]*(q.q[0]*q.q[1] + q.q[2]*q.q[3]) - mag[1],
            2*b[1]*(q.q[0]*q.q[2] + q.q[1]*q.q[3]) + 2*b[3]*(0.5 - q.q[1]**2 - q.q[2]**2) - mag[2]
        ])

        
        j = np.array([
            [-2*q.q[2],                  2*q.q[3],                  -2*q.q[0],                  2*q.q[1]],
            [2*q.q[1],                   2*q.q[0],                  2*q.q[3],                   2*q.q[2]],
            [0.,                        -4*q.q[1],                 -4*q.q[2],                  0.],
            [-2*b[3]*q.q[2],             2*b[3]*q.q[3],             -4*b[1]*q.q[2]-2*b[3]*q.q[0], -4*b[1]*q.q[3]+2*b[3]*q.q[1]],
            [-2*b[1]*q.q[3]+2*b[3]*q.q[1], 2*b[1]*q.q[2]+2*b[3]*q.q[0], 2*b[1]*q.q[1]+2*b[3]*q.q[3],  -2*b[1]*q.q[0]+2*b[3]*q.q[2]],
            [2*b[1]*q.q[2],              2*b[1]*q.q[3]-4*b[3]*q.q[1], 2*b[1]*q.q[0]-4*b[3]*q.q[2],  2*b[1]*q.q[1]]
        ])

        step = j.T.dot(f)
        step /= norm(step)  # Normalize step magnitude

        # Rate of change of q
        qgyr = Quaternion(np.array([0., gyr[0], gyr[1], gyr[2]]))
        qdot = q.prod(qgyr).q * .5 - self.beta * step.T

        # Integrate
        q.q += qdot * self.ts
        self.q0 = Quaternion( q.q / norm(q.q))

        return self.q0.q

    ## --- For Inertial Sensors --- ##
    def UpdateIMU(self, acc, gyr):
        q = self.q0
    
        # Make sure that each measure is squeezed to 1D -- (3,)
        acc = np.array(acc, dtype = float).flatten()
        gyr = np.array(gyr, dtype = float).flatten()

        # Normalize accelerometer
        acc /= norm(acc)

        # Apply gradient descent
        f = np.array([
        2*(q.q[1]*q.q[3] - q.q[0]*q.q[2]) - acc[0],
        2*(q.q[0]*q.q[1] + q.q[2]*q.q[3]) - acc[1],
        2*(0.5 - q.q[1]**2 - q.q[2]**2) - acc[2]
        ])

        j = np.array([
        [-2*q.q[2], 2*q.q[3], -2*q.q[0], 2*q.q[1]],
        [2*q.q[1], 2*q.q[0], 2*q.q[3], 2*q.q[2]],
        [0., -4*q.q[1], -4*q.q[2], 0.]
        ])

        step = j.T.dot(f)
        step /= norm(step)  # Normalize step magnitude

        qgyr = Quaternion(np.array([0., gyr[0], gyr[1], gyr[2]]))
        qdot = q.prod(qgyr).q * .5 - self.beta * step.T

        # Integrate
        q.q += qdot * self.ts
        self.q0 = Quaternion( q.q / norm(q.q))

        return self.q0.q




