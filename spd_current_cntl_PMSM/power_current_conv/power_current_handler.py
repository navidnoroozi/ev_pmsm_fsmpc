
import math
class RequiredPowerCurrentHandler:
    def __init__(self, P_req, Q_req, V_rms):
        """
        Initializes the Power and Current Handler.

        Parameters:
        - P_req: Required active power (W).
        - Q_req: Required reactive power (VAR).
        - V_rms_req: Required RMS voltage (V).
        """
        self.P=P_req; self.Q=Q_req; self.V=V_rms
    def calculateCurrentMagnitudeAndPhase(self):
        S = (self.P**2 + self.Q**2)**0.5
        I_rms = S / self.V if self.V>0 else 0.0
        I_peak = I_rms * (2**0.5)
        phi = math.atan2(self.Q, self.P) if self.P or self.Q else 0.0
        return I_peak, phi
