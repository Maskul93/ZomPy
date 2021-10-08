import numpy as np

class Quaternion: 
    def __init__(self, q = np.array([1., 0., 0., 0.])):
        self.q = q        

    ## Allows access to each quaternion component separately as q.*, where '*' could be either w, x, y, or z.
    @property
    def w(self) -> float:
        return self.q[0]

    @property
    def x(self) -> float:
        return self.q[1]

    @property
    def y(self) -> float:
        return self.q[2]

    @property
    def z(self) -> float:
        return self.q[3]

    # Print the quaternion with "print(q)" as a sum q = w + xi + yj + zk
    def __str__(self):
        return "q = [{:-.3f} {:+.3f}i {:+.3f}j {:+.3f}k]".format(self.w, self.x, self.y, self.z)
    
    # Conjugate Quaternion (q*). If q = w + xi + yj + zk --> q* = w - xi - yj - zk 
    @property
    def conj(self):
        return Quaternion(np.array([self.w, -self.x, -self.y, -self.z]))

    # Unit Quaternion
    @property
    def unit(self):
        return Quaternion( self.q / np.linalg.norm(self.q) )

    # Quaternion multiplication according to Hamiltonian product
    def prod(self, q2):
        return Quaternion(np.array([self.w*q2.w - self.x*q2.x - self.y*q2.y - self.z*q2.z,
                     self.w*q2.x + self.x*q2.w + self.y*q2.z - self.z*q2.y,
                     self.w*q2.y - self.x*q2.z + self.y*q2.w + self.z*q2.x,
                     self.w*q2.z + self.x*q2.y - self.y*q2.x + self.z*q2.w]))

    # From DCM --> Quaternion
    # See https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
    def from_DCM(R):
        return Quaternion(np.array([np.sqrt(1 + R.r11 + R.r22 + R.r33) * .5,
                            (R.r32 - R.r23) / (2 * (np.sqrt(1 + R.r11 + R.r22 + R.r33))),
                            (R.r13 - R.r31) / (2 * (np.sqrt(1 + R.r11 + R.r22 + R.r33))),
                            (R.r21 - R.r12) / (2 * (np.sqrt(1 + R.r11 + R.r22 + R.r33))) ]))