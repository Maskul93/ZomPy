import numpy as np
from scipy import constants, integrate
from beautifultable import BeautifulTable
from vmdpy import VMD
import matplotlib.pyplot as plt

## --- VMD Parameters --- ##
alpha = 100        # Mid Bandwidth Constrain  
tau = 0            # Noise-tolerance (no strict fidelity enforcement)  
K = 3              # 3 IMFs  
DC = 0             # DC part not imposed  
init = 0           # Initialize omegas uniformly  
tol = 1e-20        # Tolerance parameter

class CMJ:
    def __init__(self, a, fs = 128):
        self.a = a
        self.fs = fs
        self.ts = 1 / self.fs
        self.features = self.JumpFeatures(self)

    @property
    def t_0(self) -> int:
        ## -- 2. Unweighting Phase -- ##
        self.thr_t0 = 5 * np.std(self.a[0:99])

        # Find the first sample k : a[k+1] - a[k] > thr
        for k in range(0, self.a.shape[0] - 1):
            if (-self.a[k]  > self.thr_t0): #- self.a[k+1]
                t_0 = k - int(round((0.03 * self.fs)))
                break
        return t_0

    @property
    def v(self):
        # Compute Velocity from "onset"
        t = np.linspace(0, (self.a.shape[0] - self.t_0)/ self.fs, num = self.a.shape[0] - self.t_0)
        vt = integrate.cumtrapz(self.a[self.t_0:], t, initial = 0)
        # fill v with zeros to match a.shape
        buffer = np.zeros(self.t_0)
        return np.append(buffer, vt)
        
    @property
    def t_UB(self) -> int:
        # The end of (U) occurs when, after the Onset, the BW > 0 <==> a > 0
        for k in range(self.t_0 + 30, self.a.shape[0]):
            if (self.a[k] > 0):
                t_UB = k
                break
        return t_UB
    
    @property
    def t_BP(self) -> int:
        ## -- 3. Breaking Phase -- ##
        # Find the first sample such that V > 0
        for k in range(self.t_UB, self.a.shape[0]):
            if (self.v[k] > 0):
                t_BP = k
                break
        return t_BP

    @property
    def t_TO(self) -> int:
        ## -- 4. Propulsion Phase -- ##
        # From BP to "end", find the first k : a[k] < -g
        for k in range(self.t_BP, self.a.shape[0]):
            if (self.a[k] < - constants.g):
                t_TO = k
                break
        return t_TO

    ## -- Power -- ##
    @property
    def P(self):
        P = np.zeros(self.t_TO + 30)
        i = self.t_0
        for k in range(self.t_0, self.t_TO + 30):
            P[i] = (self.a[k] + constants.g) * self.v[k]    ## C VO A COSTANT SANG D GIUD! 
            i += 1
        return P

    @property
    def h(self):
        return (.5 * np.square(self.v[self.t_TO])) / constants.g

    def plot(self, dpi = 150):
        fig = plt.figure()
        fig.dpi = dpi
        t = np.linspace(0, self.a.shape[0] / self.fs, num = self.a.shape[0])
        plt.plot(t, self.a, color = "k", linewidth = 1)

        # Store boundaries for plotting #
        sup_lim = 10 + np.max(self.a[ : self.t_TO])
        inf_lim = self.a[self.t_TO] - 2
        arr_h = np.max(self.a[ : self.t_TO]) + 5    # Arrow height
        lab_h = arr_h + 1.3     # Label height

        ## -- Plot "pieces" of acceleration -- ##
        plt.plot(np.linspace(0, t[self.t_0], num = self.t_0), self.a[:self.t_0], color = "b", linewidth = 1.1)
        plt.plot(np.linspace(t[self.t_0], t[self.t_UB] - t[1], num = self.t_UB - self.t_0), self.a[self.t_0 : self.t_UB], color = "r", linewidth = 1.1)
        plt.plot(np.linspace(t[self.t_UB], t[self.t_BP] - t[1], num = self.t_BP - self.t_UB), self.a[self.t_UB : self.t_BP], color = "b", linewidth = 1.1)
        plt.plot(np.linspace(t[self.t_BP], t[self.t_TO], num = self.t_TO - self.t_BP + 1), self.a[self.t_BP : self.t_TO + 1], color = "r", linewidth = 1.1)
        plt.plot(np.linspace(t[self.t_TO], self.a.shape[0] - self.t_TO, num = self.a.shape[0] - self.t_TO), self.a[self.t_TO : ], color = "b", linewidth = 1.1)

        ## -- Plot vertical lines
        plt.vlines(t[self.t_UB] - t[1], -sup_lim, sup_lim + 2, colors='k', linestyles='dashed', linewidth = .75)
        plt.vlines(t[self.t_0] - t[1], -sup_lim, sup_lim + 2, colors='k', linestyles='dashed', linewidth = .75)
        plt.vlines(t[self.t_BP] - t[1], -sup_lim, sup_lim + 2, colors='k', linestyles='dashed', linewidth = .75)
        plt.vlines(t[self.t_TO], -sup_lim, sup_lim + 2, colors='k', linestyles='dashed', linewidth = .75)

        ## -- Plot arrows -- ##
        # W #
        plt.arrow(t[0], arr_h, - t[0] + t[self.t_0] - t[2], 0.0, color='black',
                head_length = .02, head_width = .5, 
                length_includes_head = True, linewidth = .8)
        plt.arrow(t[self.t_0] - t[2], arr_h, - t[self.t_0] + t[0] + t[3], 0.0, color='black',
                head_length = .02, head_width = .5, 
                length_includes_head = True, linewidth = .8)
        plt.annotate("W", xy=((t[self.t_0]) / 2, lab_h), horizontalalignment = "center")

        # U #
        plt.arrow(t[self.t_0], arr_h, - t[self.t_0] + t[self.t_UB] - t[2], 0.0, color='black',
                head_length = .02, head_width = .5, 
                length_includes_head = True, linewidth = .8)
        plt.arrow(t[self.t_UB] - t[3], arr_h, - t[self.t_UB] + t[self.t_0] + t[3], 0.0, color='black',
                head_length = .02, head_width = .5, 
                length_includes_head = True, linewidth = .8)
        plt.annotate("U", xy=((t[self.t_UB] + t[self.t_0] - t[2]) / 2, lab_h), horizontalalignment = "center")

        # B #
        plt.arrow(t[self.t_UB], arr_h, - t[self.t_UB] + t[self.t_BP] - t[2], 0.0, color='black',
                head_length = .02, head_width = .5, 
                length_includes_head = True, linewidth = .8)
        plt.arrow(t[self.t_BP] - t[3], arr_h, - t[self.t_BP] + t[self.t_UB] + t[3], 0.0, color='black',
                head_length = .02, head_width = .5, 
                length_includes_head = True, linewidth = .8)
        plt.annotate("B", xy=((t[self.t_BP] + t[self.t_UB] - t[2]) / 2, lab_h), horizontalalignment = "center")

        # P #
        plt.arrow(t[self.t_BP], arr_h, - t[self.t_BP] + t[self.t_TO] - t[1], 0.0, color='black',
                head_length = .02, head_width = .5, 
                length_includes_head = True, linewidth = .8)
        plt.arrow(t[self.t_TO] - t[3], arr_h, - t[self.t_TO] + t[self.t_BP] + t[3], 0.0, color='black',
                head_length = .02, head_width = .5, 
                length_includes_head = True, linewidth = .8)
        plt.annotate("P", xy=((t[self.t_TO] + t[self.t_BP] - t[2]) / 2, lab_h), horizontalalignment = "center")

        # F #
        plt.arrow(t[self.t_TO], arr_h, - t[self.t_TO] + t[self.t_TO + 40] - t[1], 0.0, color='black',
                head_length = .02, head_width = .5, 
                length_includes_head = True, linewidth = .8)
        plt.arrow(2.25 - t[2], arr_h, - 2.25 + t[self.t_TO] + t[3], 0.0, color='black',
                head_length = .02, head_width = .5, 
                length_includes_head = True, linewidth = .8)
        plt.annotate("F", xy=((t[self.t_TO] + t[self.t_TO + 40] - t[2]) / 2, lab_h), horizontalalignment = "center")

        plt.ylim([inf_lim, sup_lim])
        plt.ylabel("Acceleration (m/s$^2$)")
        plt.xlim([0,t[self.t_TO + 40]])
        plt.xlabel("Time (s)")

        return fig

    class JumpFeatures:            
        def __init__(self, CMJ):
            ## -- A -- ##
            self.A = (CMJ.t_UB - CMJ.t_0) / CMJ.fs

            ## -- b -- ##
            self.b = np.min(CMJ.a[CMJ.t_0 : CMJ.t_BP])

            ## -- C -- ##
            a_min = np.argmin(CMJ.a[CMJ.t_0 : CMJ.t_BP])
            a_max = np.argmax(CMJ.a[CMJ.t_0 : CMJ.t_TO])
            self.C = (a_max - a_min) / CMJ.fs

            ## -- D -- ##
            for k in range(CMJ.t_UB, CMJ.t_TO):
                if CMJ.a[k] < 0:
                    F_0 = k-1
                    break

            self.D = (F_0 - CMJ.t_UB) / CMJ.fs

            ## -- e -- ##
            self.e = np.max(CMJ.a[CMJ.t_0 : CMJ.t_TO])

            ## -- F -- ##
            self.F = (CMJ.t_TO - a_max) / CMJ.fs

            ## -- G -- ##
            self.G = (CMJ.t_TO - CMJ.t_0) / CMJ.fs

            ## -- H -- ##
            self.H = (CMJ.t_BP - a_min) / CMJ.fs

            ## -- i -- ##
            tilt = np.diff(CMJ.a[a_min : a_max + 1])
            self.i = CMJ.a[CMJ.t_0 + a_min + np.argmax(tilt)]

            ## -- J -- ##
            v_min = np.argmin(CMJ.v[ : CMJ.t_BP])
            self.J = (CMJ.t_BP - v_min) / CMJ.fs

            ## -- k -- ##
            self.k = CMJ.a[CMJ.t_BP]

            ## -- l -- ##
            self.l = np.min(CMJ.P[CMJ.t_UB : CMJ.t_BP])

            ## -- M -- ##
            for k in range(CMJ.t_BP + 3, CMJ.P.shape[0]):
                if CMJ.P[k] < 0:
                    P_0 = k-1
                    break

            self.M = (P_0 - CMJ.t_BP) / CMJ.fs

            ## -- n -- ##
            self.n = np.max(CMJ.P)

            ## -- O -- ##
            P_max = np.argmax(CMJ.P)
            self.O = (CMJ.t_TO - P_max) / CMJ.fs

            ## -- p -- ##
            self.p = (self.e - self.b) / self.C

            ## -- q -- ##
            shape = integrate.trapz(CMJ.a[CMJ.t_UB : F_0], dx = 1 / CMJ.fs)
            self.q = shape / (self.D * self.e)

            ## -- r -- ##
            self.r = self.b / self.e

            ## -- s -- ##
            self.s = np.min(CMJ.v[ : np.argmax(CMJ.v)])

            ## -- t -- ##
            self.t = np.mean(CMJ.P[CMJ.t_0 : CMJ.t_BP])

            ## -- u -- ##
            self.u = np.mean(CMJ.P[CMJ.t_BP : CMJ.t_TO + 1])

            ## -- W -- ##
            self.W = ( np.argmax(CMJ.P) - np.argmin(CMJ.P[ : np.argmax(CMJ.P)]) ) / CMJ.fs

            ## -- f1, f2, f3 -- ##
            u, u_hat, omega = VMD(CMJ.a, alpha, tau, K, DC, init, tol)  
            self.f3, self.f2, self.f1 = omega[-1] * ( CMJ.fs / 2 ) 

            # -- Store IMFs -- ##
            self.u3, self.u2, self.u1 = u

        def display(self, CMJ):
            table = BeautifulTable(maxwidth = 250)
            table.columns.header = ["A", "b", "C", "D", "e", "F", 
                "G", "h", "i", "J", "k", "l", "M", "n", "O", "p", 
                "q", "r", "s", "t", "u", "W", "f1", "f2", "f3", "height"]
            table.rows.append([self.A, self.b, self.C, self.D, 
                self.e, self.F, self.G, self.H, self.i, 
                self.J, self.k, self.l, self.M, self.n, 
                self.O, self.p, self.q, self.r, self.s, 
                self.t, self.u, self.W, self.f1,
                self.f2, self.f3, CMJ.h])
            return table
            
        def get_them(self, CMJ):
            return [self.A, self.b, self.C, self.D, 
                self.e, self.F, self.G, self.H, self.i, 
                self.J, self.k, self.l, self.M, self.n, 
                self.O, self.p, self.q, self.r, self.s, 
                self.t, self.u, self.W, self.f1,
                self.f2, self.f3, CMJ.h]

        def plot(self, CMJ, dpi = 150):

            ## Acceleration Features ##
            fig1 = plt.figure()
            fig1.dpi = dpi
            a_0 = int(self.D * CMJ.fs + CMJ.t_UB) # When the acc goes to 0 prior "t_TO"
            P_max = np.argmax(CMJ.P) # Self explanatory
            
            plt.plot(list(range(0, CMJ.a.shape[0])), np.zeros( CMJ.a.shape[0]) + .1, "--", color = "k", 
                linewidth = .5)

            ## -- Plot rectangle of area D*e -- ##
            plt.vlines(CMJ.t_UB, 0, self.e + .1, colors='k', linestyles='dashed', linewidth = 1)
            plt.vlines(a_0 + .5, 0, self.e + .1, colors='k', linestyles='dashed', linewidth = 1)
            plt.plot(list(range(CMJ.t_UB, a_0 + 1)), (self.e + .1) * np.ones(a_0 - CMJ.t_UB + 1), "--", color = "k", linewidth = 1)

            # # ## -- Plot the signal and highlight positive and negative portion of it -- ##
            plt.plot(CMJ.a, color = "k", linewidth = 1)
            plt.fill_between(list(range(CMJ.t_0 + 3, CMJ.t_UB)), CMJ.a[list(range(CMJ.t_0 + 3, CMJ.t_UB))], alpha = .25, color = "r")
            plt.fill_between(list(range(CMJ.t_UB, a_0 + 1)), CMJ.a[list(range(CMJ.t_UB, a_0 + 1))], alpha = .25)
              

            ## -- A -- ##
            plt.arrow(CMJ.t_0 + 4, 1, CMJ.t_UB - CMJ.t_0 - 4.5, 0.0, color='black',
                    head_length = 2.0, head_width = .5, 
                    length_includes_head = True)
            plt.arrow(CMJ.t_UB, 1, CMJ.t_0 - CMJ.t_UB, 0.0, color='black',
                    head_length = 2.0, head_width = .5, 
                    length_includes_head = True)
            plt.annotate("A", xy=((CMJ.t_0 + CMJ.t_UB) / 2, 2.3), horizontalalignment = "center")

            ## -- C -- ##
            plt.arrow(np.argmin(CMJ.a[ : CMJ.t_UB]), self.b - 1.5, np.argmax(CMJ.a[ : CMJ.t_TO]) - np.argmin(CMJ.a[ : CMJ.t_UB]), 0.0, color='black',
                    head_length = 2.0, head_width = .5, 
                    length_includes_head = True)
            plt.arrow(np.argmax(CMJ.a[ : CMJ.t_TO]), self.b - 1.5, np.argmin(CMJ.a[ : CMJ.t_UB]) - np.argmax(CMJ.a[ : CMJ.t_TO]), 0.0, color='black',
                    head_length = 2.0, head_width = .5, 
                    length_includes_head = True)
            plt.annotate("C", xy=((np.argmin(CMJ.a[ : CMJ.t_UB]) + np.argmax(CMJ.a[ : CMJ.t_TO])) / 2, self.b - 2.8), horizontalalignment = "center")

            ## -- D -- ##
            plt.arrow(CMJ.t_UB, -1, a_0 - CMJ.t_UB, 0.0, color='black',
                    head_length = 2.0, head_width = .5, 
                    length_includes_head = True)
            plt.arrow(a_0, -1, - a_0 + CMJ.t_UB, 0.0, color='black',
                    head_length = 2.0, head_width = .5, 
                    length_includes_head = True)
            plt.annotate("D", xy=((a_0 + CMJ.t_UB) / 2, -2.3), horizontalalignment = "center")

            ## -- F -- ##
            plt.arrow(P_max, -10, CMJ.t_TO - P_max, 0.0, color='black',
                    head_length = 2.0, head_width = .5, 
                    length_includes_head = True)
            plt.arrow(CMJ.t_TO, -10, P_max - CMJ.t_TO, 0.0, color='black',
                    head_length = 2.0, head_width = .5, 
                    length_includes_head = True)
            plt.annotate("F", xy=((P_max + CMJ.t_TO) / 2, -11.3), horizontalalignment = "center")

            # ## -- G -- ##

            plt.arrow(CMJ.t_0, np.max(CMJ.a[ : CMJ.t_TO]) + 1, CMJ.t_TO - CMJ.t_0, 0.0, color='black',
                    head_length = 2.0, head_width = .5, 
                    length_includes_head = True)
            plt.arrow(CMJ.t_TO, np.max(CMJ.a[ : CMJ.t_TO]) + 1, CMJ.t_0 - CMJ.t_TO, 0.0, color='black',
                    head_length = 2.0, head_width = .5, 
                    length_includes_head = True)
            plt.annotate("G", xy=((CMJ.t_0 + CMJ.t_TO) / 2, np.max(CMJ.a[ : CMJ.t_TO]) + 2.3), horizontalalignment = "center")

            # ## -- H -- ##
            plt.arrow(np.argmin(CMJ.a[ : CMJ.t_UB]), self.b - 3.5, CMJ.t_BP - np.argmin(CMJ.a[ : CMJ.t_UB]), 0.0, color='black',
                    head_length = 2.0, head_width = .5, 
                    length_includes_head = True)
            plt.arrow(CMJ.t_BP, self.b - 3.5, np.argmin(CMJ.a[ : CMJ.t_UB]) - CMJ.t_BP, 0.0, color='black',
                    head_length = 2.0, head_width = .5, 
                    length_includes_head = True)
            plt.annotate("H", xy=((np.argmin(CMJ.a[ : CMJ.t_UB]) + CMJ.t_UB) / 2, self.b - 4.8), horizontalalignment = "center")

            # ## -- J -- ##
            plt.arrow(CMJ.t_UB, -4, CMJ.t_BP - CMJ.t_UB, 0.0, color='black',
                    head_length = 2.0, head_width = .5, 
                    length_includes_head = True)
            plt.arrow(CMJ.t_BP, -4, CMJ.t_UB - CMJ.t_BP, 0.0, color='black',
                    head_length = 2.0, head_width = .5, 
                    length_includes_head = True)
            plt.annotate("J", xy=((CMJ.t_BP + CMJ.t_UB) / 2, -5.3), horizontalalignment = "center")


            # ## -- Single points: b, e, (i), k -- ##
            plt.plot(np.argmin(CMJ.a[ : CMJ.t_UB]), CMJ.a[np.argmin(CMJ.a[ : CMJ.t_UB])], "*", color = "k")
            plt.plot(np.argmax(CMJ.a[ : CMJ.t_TO]), CMJ.a[np.argmax(CMJ.a[ : CMJ.t_TO])], "*", color = "k")
            plt.plot(CMJ.t_BP, CMJ.a[CMJ.t_BP], "*", color = "k")

            plt.annotate("b", xy=(np.argmin(CMJ.a[ : CMJ.t_UB]), CMJ.a[np.argmin(CMJ.a[ : CMJ.t_UB])] + .8), horizontalalignment = "center")
            plt.annotate("e", xy=(np.argmax(CMJ.a[ : CMJ.t_TO]), CMJ.a[np.argmax(CMJ.a[ : CMJ.t_TO])] - 1.3), horizontalalignment = "center")
            plt.annotate("k", xy=(CMJ.t_BP, CMJ.a[CMJ.t_BP] - 1.5), horizontalalignment = "center")

            y_lim = np.max(CMJ.a[ : CMJ.t_TO]) + 6
            plt.xlim([CMJ.t_0 - 30, CMJ.t_TO + 30])
            plt.ylim([-y_lim, y_lim])
            plt.xlabel("Time (s)")
            plt.ylabel("Acceleration (m/s$^2$)")

            plt.xticks([0+CMJ.t_0-30, CMJ.fs/4+CMJ.t_0-30, CMJ.fs/2+CMJ.t_0-30, 3*CMJ.fs/4+CMJ.t_0-30, CMJ.fs+CMJ.t_0-30, 5*CMJ.fs/4+CMJ.t_0-30], ["0", "0.25", "0.5", "0.75", "1.0", "1.25"])

            fig2 = plt.figure()
            fig2.dpi = dpi

            # Store boundaries for plotting #
            sup_lim = 10 + np.max(CMJ.P)
            inf_lim = np.min(CMJ.P) - 5
            arr_h = np.max(CMJ.P) + 5    # Arrow height
            lab_h = arr_h + 1.3     # Label height

            ## -- Zero dashed line -- ##
            plt.plot(list(range(0, CMJ.a.shape[0] )), np.zeros( CMJ.a.shape[0] ), "--", color = "k", 
                linewidth = .5)

            ## -- Plot the signal and highlight positive and negative portion of it -- ##
            plt.plot(CMJ.P, linewidth = 1, color = "k")
            plt.fill_between(list(range(CMJ.t_0, CMJ.t_BP)), CMJ.P[list(range(CMJ.t_0, CMJ.t_BP))], alpha = .25)
            plt.fill_between(list(range(CMJ.t_BP - 1, CMJ.t_TO + 1)), CMJ.P[list(range(CMJ.t_BP - 1, CMJ.t_TO + 1))], alpha = .25, color = "r")

            ## -- M -- ##
            plt.arrow(CMJ.t_BP, -4, CMJ.t_TO - CMJ.t_BP, 0.0, color='black',
                    head_length = 2.0, head_width = 1, 
                    length_includes_head = True)
            plt.arrow(CMJ.t_TO, -4, CMJ.t_BP - CMJ.t_TO, 0.0, color='black',
                    head_length = 2.0, head_width = 1, 
                    length_includes_head = True)
            plt.annotate("M", xy=((CMJ.t_BP + CMJ.t_TO) / 2, -3.3), horizontalalignment = "center")

            ## -- O -- ##
            plt.arrow(P_max, arr_h, CMJ.t_TO  - P_max, 0.0, color='black',
                    head_length = 2.0, head_width = 1, 
                    length_includes_head = True)
            plt.arrow(CMJ.t_TO, arr_h, - CMJ.t_TO + P_max, 0.0, color='black',
                    head_length = 2.0, head_width = 1, 
                    length_includes_head = True)
            plt.annotate("O", xy=((P_max + CMJ.t_TO) / 2, lab_h), horizontalalignment = "center")

            ## -- W -- ##
            plt.arrow(np.argmin(CMJ.P), arr_h, P_max - np.argmin(CMJ.P) - 1, 0.0, color='black',
                    head_length = 2.0, head_width = 1, 
                    length_includes_head = True)
            plt.arrow(P_max - 1, arr_h, - P_max + np.argmin(CMJ.P), 0.0, color='black',
                    head_length = 2.0, head_width = 1, 
                    length_includes_head = True)
            plt.annotate("W", xy=((P_max + np.argmin(CMJ.P)) / 2, lab_h), horizontalalignment = "center")

            ## -- Single points: l, n -- ##
            plt.plot(np.argmin(CMJ.P), CMJ.P[np.argmin(CMJ.P)], "*", color = "k")
            plt.plot(P_max, CMJ.P[P_max], "*", color = "k")


            plt.annotate("l", xy=(np.argmin(CMJ.P), CMJ.P[np.argmin(CMJ.P)]-3.8), horizontalalignment = "center")
            plt.annotate("n", xy=(P_max - 6, CMJ.P[P_max]-.8), horizontalalignment = "center")
            plt.annotate("t", xy=( np.argmin(CMJ.P), np.min(CMJ.P) / 2), horizontalalignment = "center")
            plt.annotate("u", xy=(P_max - 6, np.max(CMJ.P) / 2), horizontalalignment = "center")


            plt.xlim([CMJ.t_0 - 30, CMJ.t_TO + 30])
            plt.ylim([inf_lim, sup_lim])
            plt.xlabel("Time (s)")
            plt.ylabel("Power (W / Body Mass)")

            plt.xticks([0 + CMJ.t_0-30, CMJ.fs/ 4 + CMJ.t_0 - 30,
                CMJ.fs / 2 + CMJ.t_0 - 30, 3 * CMJ.fs / 4 + CMJ.t_0 - 30,
                CMJ.fs + CMJ.t_0 - 30, 5 * CMJ.fs / 4 + CMJ.t_0 - 30], ["0", "0.25", "0.5", "0.75", "1.0", "1.25"])

            return fig1, fig2