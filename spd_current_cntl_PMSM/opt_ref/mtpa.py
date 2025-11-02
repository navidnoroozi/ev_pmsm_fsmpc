
# === two_level_inverter_fsmpc/opt_ref/mtpa.py ===
import math

class MTPA:
    """
    Maximum Torque per Ampere (MTPA) reference generator.

    Computes optimal (i_d_ref, i_q_ref) for a given torque or torque-producing current command.
    Can be used between the SpeedController and CurrentReference modules in FOC-FS-MPC.
    """

    def __init__(self, L_d=0.008, L_q=0.012, psi_f=0.1, pole_pairs=4):
        self.Ld = L_d
        self.Lq = L_q
        self.psi_f = psi_f
        self.p = pole_pairs

    def compute(self, torque_ref):
        """Compute optimal (i_d_ref, i_q_ref) given torque reference."""
        Ld, Lq, ψf, p = self.Ld, self.Lq, self.psi_f, self.p
        i_q_nom = torque_ref / (1.5 * p * ψf)
        ΔL = Lq - Ld
        if abs(ΔL) < 1e-9:
            return 0.0, i_q_nom
        i_d = (-ψf + math.sqrt(ψf**2 + 8*(ΔL**2)*(i_q_nom**2))) / (4*ΔL)
        i_q = i_q_nom
        return i_d, i_q

    def compute_from_iq(self, i_q_cmd):
        """Alternative: given i_q_cmd from speed controller."""
        Ld, Lq, ψf = self.Ld, self.Lq, self.psi_f
        ΔL = Lq - Ld
        if abs(ΔL) < 1e-9:
            return 0.0, i_q_cmd
        i_d = (-ψf + math.sqrt(ψf**2 + 8*(ΔL**2)*(i_q_cmd**2))) / (4*ΔL)
        return i_d, i_q_cmd
