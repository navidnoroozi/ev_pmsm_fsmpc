
class DriverPI:
    """Speed tracking PI to produce omega_m_ref from vehicle speed reference."""
    def __init__(self, Kp=1.5, Ki=0.5, Ts=0.01):
        self.Kp=Kp; self.Ki=Ki; self.Ts=Ts; self.xi=0.0

    def step(self, v_ref, v_meas, veh):
        e = v_ref - v_meas
        self.xi += e * self.Ts
        omega_m_ref = veh.G * (self.Kp*e + self.Ki*self.xi)
        return max(0.0, omega_m_ref), e