class SpeedController:
    """ Outer PI speed controller for PMSM FOC """

    def __init__(self, Kp=5.0, Ki=100.0, Ts=1e-4, i_q_ref_max=20.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Ts = Ts
        self.i_q_ref_max = i_q_ref_max
        self._int_err = 0.0

    def compute_iq_ref(self, omega_ref, omega_meas):
        e = omega_ref - omega_meas
        self._int_err += e * self.Ts
        i_q_ref = self.Kp * e + self.Ki * self._int_err
        i_q_ref = max(-self.i_q_ref_max, min(self.i_q_ref_max, i_q_ref))
        return i_q_ref, e

    def reset(self):
        self._int_err = 0.0
