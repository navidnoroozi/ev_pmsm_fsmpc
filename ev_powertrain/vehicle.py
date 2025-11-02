
import math

class Vehicle1D:
    """
    Simple longitudinal point-mass vehicle model.
    States: v [m/s]
    Parameters:
      m: mass [kg]
      Cr: rolling resistance coefficient [-]
      rho: air density [kg/m^3]
      CdA: drag area (Cd*A) [m^2]
      Rw: wheel radius [m]
      G: overall gear ratio (motor_speed = G * wheel_speed)
      eta: drivetrain efficiency (0..1)
      grade: road grade angle [rad]
    """
    def __init__(self, m=1600.0, Cr=0.010, rho=1.225, CdA=0.65, Rw=0.31, G=9.0, eta=0.95, grade=0.0):
        self.m=m; self.Cr=Cr; self.rho=rho; self.CdA=CdA; self.Rw=Rw; self.G=G; self.eta=eta; self.grade=grade
        self.v = 0.0

    def forces(self):
        F_roll = self.m * 9.81 * self.Cr * math.cos(self.grade)
        F_aero = 0.5 * self.rho * self.CdA * self.v*self.v
        F_grade = self.m * 9.81 * math.sin(self.grade)
        return F_roll, F_aero, F_grade, sum([F_roll, F_aero, F_grade])

    def v_next(self, Tm, Ts):
        """
        Apply motor torque Tm [N*m] at motor shaft; converts to wheel force via gear & efficiency.
        Updates vehicle speed v.
        """
        # Wheel torque and traction force
        Tw = self.eta * self.G * Tm
        F_trac = Tw / self.Rw

        _, _, _, F_res = self.forces()
        a = (F_trac - F_res) / self.m
        self.v = max(0.0, self.v + a * Ts)
        return self.v, a

    def motor_speed(self):
        # wheel speed = v/Rw, motor speed = G * wheel speed
        omega_w = self.v / self.Rw
        return self.G * omega_w
