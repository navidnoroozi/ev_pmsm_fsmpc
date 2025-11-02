import math
import numpy as np

class PMSMLoad:
    """ PMSM Load model (electrical + mechanical) for FS-MPC FOC system """

    def __init__(self,
                 R_s=0.5,
                 L_d=0.008,
                 L_q=0.008,
                 psi_f=0.1,
                 pole_pairs=4,
                 J=0.001,
                 B=0.0001,
                 T_L=0.0
                 ):
        self.R = float(R_s)
        self.L_d = float(L_d)
        self.L_q = float(L_q)
        self.psi_f = float(psi_f)
        self.p = int(pole_pairs)
        self.J = float(J)
        self.B = float(B)
        self.T_L = float(T_L)
        self.omega_m = 0.0
        self.theta_e = 0.0

    def _electrical_step_3phase(self, v_a, v_b, v_c, i_a, i_b, i_c, Ts):
        """
        Electrical step for 3-phase PMSM
        Inputs:
            v_a, v_b, v_c : phase voltages
            i_a, i_b, i_c : phase currents
            Ts : sampling time
        Outputs:
            i_a_next, i_b_next, i_c_next : next phase currents
            i_d_next, i_q_next : next dq currents
        """
        R = self.R
        Ld, Lq = self.L_d, self.L_q
        psi_f = self.psi_f
        w_e = self.p * self.omega_m
        theta_e = self.theta_e

        # Clarke and Park transformations
        i_alpha = (2/3)*(i_a - 0.5*i_b - 0.5*i_c)
        i_beta  = (2/3)*(math.sqrt(3)/2)*(i_b - i_c)
        v_alpha = (2/3)*(v_a - 0.5*v_b - 0.5*v_c)
        v_beta  = (2/3)*(math.sqrt(3)/2)*(v_b - v_c)

        v_d =  math.cos(theta_e)*v_alpha + math.sin(theta_e)*v_beta
        v_q = -math.sin(theta_e)*v_alpha + math.cos(theta_e)*v_beta
        i_d =  math.cos(theta_e)*i_alpha + math.sin(theta_e)*i_beta
        i_q = -math.sin(theta_e)*i_alpha + math.cos(theta_e)*i_beta

        # PMSM dq equations
        di_d = (v_d - R*i_d + w_e*Lq*i_q) / Ld
        di_q = (v_q - R*i_q - w_e*(Ld*i_d + psi_f)) / Lq

        i_d_next = i_d + di_d * Ts
        i_q_next = i_q + di_q * Ts

        # inverse Park transform
        i_alpha_next = math.cos(theta_e)*i_d_next - math.sin(theta_e)*i_q_next
        i_beta_next  = math.sin(theta_e)*i_d_next + math.cos(theta_e)*i_q_next

        i_a_next = i_alpha_next
        i_b_next = -0.5*i_alpha_next + (math.sqrt(3)/2)*i_beta_next
        i_c_next = -0.5*i_alpha_next - (math.sqrt(3)/2)*i_beta_next

        return i_a_next, i_b_next, i_c_next, i_d_next, i_q_next

    def step(self, v_a, v_b, v_c, i_a, i_b, i_c, Ts):
        """
        Perform a single simulation step
        """
        i_a_next, i_b_next, i_c_next, i_d_next, i_q_next = self._electrical_step_3phase(
            v_a, v_b, v_c, i_a, i_b, i_c, Ts
        )
        T_e = 1.5 * self.p * (self.psi_f * i_q_next + (self.L_d - self.L_q)*i_d_next*i_q_next)
        domega = (T_e - self.T_L - self.B*self.omega_m) / self.J
        self.omega_m += domega * Ts
        self.theta_e += self.p * self.omega_m * Ts
        self.theta_e = math.fmod(self.theta_e, 2*math.pi)
        return (i_a_next, i_b_next, i_c_next), (i_d_next, i_q_next), T_e, self.omega_m

    def get_theta_e(self): return self.theta_e
    def get_omega_m(self): return self.omega_m