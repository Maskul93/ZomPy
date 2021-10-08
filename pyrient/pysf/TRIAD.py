"""
TRIAD 
=====

This method estimates the 3D MIMU orientation taking into account a combination of the *gravity* and *magnetic North* vectory. The orientation computed through it is expressed as the North-East-Down (NED) type. 
It requires both accelerometer and magnetometer measures, and its usage is recommended to estimate static orientation (better if just initial condition), rather than dynamic ones. 

Algorithm
=========
Measurement normalization
=========================
Let consider the accelerometer and magnetometer measures, :math:`\\mathbf{a}` and `\\mathbf{h}`
"""

from ..orientclasses import Quaternion as Q
from ..orientclasses import DCM as RotM
import numpy as np

# Unit Vector
def unit(x):
    return x / np.linalg.norm(x)

## -- TRIAD algorithm -- ##
def TRIAD(a, h, output = "q"):

    """TRIAD algorithm for attitude and heading estimate.

    It estimates the orientation of a Magnetic and Inertial Measurement Unit (MIMU) exploiting accelerometer and magnetometer measures. This implementation is irrespective of the measure unit of the two measures, being both normalized later.

    Parameters
    ----------
    Required:
        - a : numpy.ndarray 
            - accelerometer measure 
        - h : numpy.ndarray
            - magnetometer measure

    Optional:
        - output = "q" : string
            - It can be either "q" (quaternion) or "DCM" (Direction Cosine Matrix)

    Returns
    -------
        - q (or R) : Quaternion (DCM) object
            - The orientation estimate after applying TRIAD algorithm.
    
    """

    a_t = unit(a)
    h_t = unit(h)

    a_n = np.array([0, 0, 1])

    h_n_tmp = np.array([np.linalg.norm( np.array([h_t[0], -h_t[1]])),  
                                        0, 
                                        -h_t[2]])
    h_n = unit(h_n_tmp)

    r1 = a_n
    r2 = unit( np.cross(r1, h_n) )
    r3 = np.cross(r1, r2)

    gRn = RotM.DCM(np.matrix([r1, r2, r3]))
    
    s1 = a_t
    s2 = np.cross(s1, h_t) / np.linalg.norm( np.cross(s1, h_t) )
    s3 = np.cross(s1, s2)
    gRt = RotM.DCM(np.matrix([s1, s2, s3]))
    tRg = gRt.transp

    tRn = tRg.prod(gRn)
    
    if output == "q":
        return Q.Quaternion.from_DCM(tRn) 
    if output == "DCM":
        return tRn