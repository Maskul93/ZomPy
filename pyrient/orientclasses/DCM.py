import numpy as np
import matplotlib.pyplot as plt

class DCM:
    def __init__(self, R):
        self.R = R
        
        # Check for orthonormality
        #if np.abs(np.linalg.det(self.R)) != 1:
        #    print("WARNING! Your DCM is NOT orthogonal")

    # Single elements
    @property
    def r11(self) -> float:
        return self.R[0,0]

    @property
    def r12(self) -> float:
        return self.R[0,1]

    @property
    def r13(self) -> float:
        return self.R[0,2]

    @property
    def r21(self) -> float:
        return self.R[1,0]

    @property
    def r22(self) -> float:
        return self.R[1,1]

    @property
    def r23(self) -> float:
        return self.R[1,2]

    @property
    def r31(self) -> float:
        return self.R[2,0]

    @property
    def r32(self) -> float:
        return self.R[2,1]

    @property
    def r33(self) -> float:
        return self.R[2,2]
    
    # Print the DCM easily
    def __str__(self):
        return ("    |{:+.3f} {:+.3f} {:+.3f}|\n".format(self.r11, self.r12, self.r13)  +
                "R = |{:+.3f} {:+.3f} {:+.3f}|\n".format(self.r21, self.r22, self.r23) +  
                "    |{:+.3f} {:+.3f} {:+.3f}|\n\n".format(self.r31, self.r32, self.r33))

    def plot(self, col = "r"):

        ax = plt.figure().add_subplot(projection = '3d')
        ax.set_xlim3d(-1.01, 1.01)
        ax.set_ylim3d(-1.01, 1.01)
        ax.set_zlim3d(-1.01, 1.01)

        x, y, z = 0, 0, 0
        v1 = np.array([self.r11, self.r21, self.r31])
        v2 = np.array([self.r12, self.r22, self.r32])
        v3 = np.array([self.r13, self.r23, self.r33])

        ax.quiver(x, y, z, v1, v2, v3, length = 1, normalize = True,
                    color = col)

        plt.show()
    
    # Transpose 
    @property
    def transp(self):
        return DCM(self.R.T)

    # DCM product
    def prod(self, R2):
        return DCM(np.matmul(self.R, R2.R))

    def to_EUL(self, seq = "XYZ", mu = "rad"):

        if seq == "XYZ":
            t1 = np.arctan( - self.r23 / self.r33 )
            t2 = np.arctan( self.r13 / np.sqrt( 1 - np.square(self.r13) ) )
            t3 = np.arctan( -self.r12 / self.r11 )
        

        if seq == "XZY":
            t1 = np.arctan( self.r32 / self.r22 )
            t2 = np.arctan( self.r12 / np.sqrt( 1 - np.square(self.r12) ) )
            t3 = np.arctan( self.r13 / self.r11 )
        
        if seq == "XYX":
            t1 = np.arctan( self.r21 / -self.r31 )
            t2 = np.arctan( np.sqrt( 1 - np.square(self.r11) ) / self.r11 )
            t3 = np.arctan( self.r12 / self.r13 )

        if seq == "XZX":
            t1 = np.arctan( self.r31 / self.r21 )
            t2 = np.arctan( np.sqrt( 1 - np.square(self.r11) ) / self.r11 )
            t3 = np.arctan( self.r13 / -self.r12 )

        if seq == "YXZ":
            t1 = np.arctan( self.r31 / self.r33 )
            t2 = np.arctan( -self.r23 / np.sqrt( 1 - np.square(self.r23) ) )
            t3 = np.arctan( self.r21 / self.r22 )

        if seq == "YZX":
            t1 = np.arctan( -self.r31 / self.r11 )
            t2 = np.arctan( self.r21 / np.sqrt( 1 - np.square(self.r21) ) )
            t3 = np.arctan( -self.r23 / self.r22 )

        if seq == "YXY":
            t1 = np.arctan( self.r12 / self.r32 )
            t2 = np.arctan( np.sqrt( 1 - np.square(self.r22) ) / self.r22 )
            t3 = np.arctan( self.r21 / -self.r23 )

        if seq == "YZY":
            t1 = np.arctan( self.r32 / -self.r12 )
            t2 = np.arctan( np.sqrt( 1 - np.square(self.r22) ) / self.r22 )
            t3 = np.arctan( self.r23 / self.r21 )

        if seq == "ZXY":
            t1 = np.arctan( -self.r12 / self.r22 )
            t2 = np.arctan( self.r32 / np.sqrt( 1 - np.square(self.r32) ) )
            t3 = np.arctan( -self.r31 / self.r33 )

        if seq == "ZYX":
            t1 = np.arctan( self.r21 / self.r11 )
            t2 = np.arctan( -self.r31 / np.sqrt( 1 - np.square(self.r31) ) )
            t3 = np.arctan( self.r32 / self.r33 )

        if seq == "ZXZ":
            t1 = np.arctan( self.r13 / -self.r23 )
            t2 = np.arctan( np.sqrt( 1 - np.square(self.r33) ) / self.r33 )
            t3 = np.arctan( self.r31 / self.r32 )

        if seq == "ZYZ":
            t1 = np.arctan( self.r23 / self.r13 )
            t2 = np.arctan( np.sqrt( 1 - np.square(self.r33) ) / self.r33 )
            t3 = np.arctan( self.r32 / -self.r31 )

        if mu == "rad":
            return np.array([t1, t2, t3])
        if mu == "deg":
            return np.array([t1, t2, t3])*180/np.pi

    # From Quaternion --> DCM
    def from_q(q):
        return DCM(np.matrix([[np.square(q.w) + np.square(q.x) - np.square(q.y) - np.square(q.z),
            2 * (q.x * q.y - q.w * q.z),
            2 * (q.x * q.z + q.w * q.y)],
            [2 * (q.x * q.y + q.w * q.z),
            np.square(q.w) - np.square(q.x) + np.square(q.y) - np.square(q.z),
            2 * (q.y * q.z - q.w * q.x)],
            [2 * (q.x * q.z - q.w * q.y),
            2 * (q.y * q.z + q.w * q.x),
            np.square(q.w) - np.square(q.x) - np.square(q.y) + np.square(q.z)]]))