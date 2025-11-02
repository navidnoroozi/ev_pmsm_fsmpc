import math
import numpy as np

class CurrentReference:
    """ dq→αβ reference converter for PMSM FOC-FS-MPC """

    def __init__(self, sampling_rate, cont_horizon):
        self.sampling_rate = sampling_rate
        self.cont_horizon = cont_horizon

    def set_dq_refs(self, i_d_ref, i_q_ref, theta_e):
        c, s = math.cos(theta_e), math.sin(theta_e)
        i_alpha_ref = c*i_d_ref - s*i_q_ref
        i_beta_ref  = s*i_d_ref + c*i_q_ref
        self.i_alpha_ref = i_alpha_ref
        self.i_beta_ref = i_beta_ref
        return i_alpha_ref, i_beta_ref

    def get_alpha_beta_refs(self):
        return self.i_alpha_ref, self.i_beta_ref

    def generateThreePhaseRefs(self, theta_e, i_d_ref, i_q_ref):
        c, s = math.cos(theta_e), math.sin(theta_e)
        i_alpha = c*i_d_ref - s*i_q_ref
        i_beta  = s*i_d_ref + c*i_q_ref
        i_a = i_alpha
        i_b = -0.5*i_alpha + (math.sqrt(3)/2)*i_beta
        i_c = -0.5*i_alpha - (math.sqrt(3)/2)*i_beta
        return np.array([i_a]), np.array([i_b]), np.array([i_c])