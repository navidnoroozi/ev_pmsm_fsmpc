# === mpc_contr_calc.py (Unified FS-MPC) ===
import math
import itertools
import numpy as np

# ---------------- Clarke (power-invariant, balanced) ----------------
SQ23  = math.sqrt(2.0/3.0)
SQ3_2 = math.sqrt(3.0)/2.0

def clarke_power_invariant(a, b, c):
    """
    Vectorized, power-invariant Clarke transform for arrays a,b,c (same length).
    Returns (i_alpha, i_beta).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    i_alpha = SQ23 * (a - 0.5*b - 0.5*c)
    i_beta  = SQ23 * (SQ3_2 * (b - c))
    return i_alpha, i_beta

# -------------------------- MPC Solver ------------------------------
class MPCSSolver:
    """
    Unified FS-MPC:
      Primary   : brute-force (single-phase: 2^N; three-phase: 8^N up to N<=3)
      Optional  : Gurobi MIQP (if use_gurobi=True)
      Fallback  : per-phase PI safe mode (and last-action reuse)

    For large three-phase horizons without Gurobi, a BEAM SEARCH is used
    to maintain look-ahead and prevent performance degradation as N grows.

    Cost (three-phase):
        J = Σ_k [ (iα[k]-iαref[k])² + (iβ[k]-iβref[k])² ]
            + λ_sw Σ_k [ (Δsa)² + (Δsb)² + (Δsc)² ]
    """

    def __init__(self,
                 cont_horizon=1,
                 lambda_switch=0.1,
                 use_gurobi=False,
                 timelimit=0.002,
                 mipgap=0.05,
                 threads=1,
                 safe_Kp=0.2,
                 safe_Ki=50.0):
        self.cont_horizon  = int(cont_horizon)
        self.lambda_switch = float(lambda_switch)
        self.use_gurobi    = bool(use_gurobi)
        self.timelimit     = float(timelimit)
        self.mipgap        = float(mipgap)
        self.threads       = int(threads)
        self.safe_Kp       = float(safe_Kp)
        self.safe_Ki       = float(safe_Ki)

        # safe-mode PI states
        self._int_err_a = 0.0
        self._int_err_b = 0.0
        self._int_err_c = 0.0
        self._last_action = 1

        # choices for 3-phase brute force
        self._abc_set = [(0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)]

        # try import gurobi lazily
        try:
            import gurobipy as gp  # noqa: F401
            self._has_grb = True
        except Exception:
            self._has_grb = False

    # ---------------------- cost helpers -----------------
    def _cost_three(self, ia_pred, ib_pred, ic_pred, ia_ref, ib_ref, ic_ref, s_seq, s0):
        iα_pred, iβ_pred = clarke_power_invariant(ia_pred, ib_pred, ic_pred)
        iα_ref,  iβ_ref  = clarke_power_invariant(ia_ref,  ib_ref,  ic_ref)
        track = np.sum((iα_pred - iα_ref)**2 + (iβ_pred - iβ_ref)**2)
        sw = 0.0
        for k,(sa,sb,sc) in enumerate(s_seq):
            if k == 0 and s0:
                psa, psb, psc = s0[0]
            else:
                psa, psb, psc = s_seq[k-1]
            sw += (sa-psa)**2 + (sb-psb)**2 + (sc-psc)**2
        return float(track + self.lambda_switch * sw)

    # ---------------------- brute force ------------------
    def _bf_three(self, inverter, load, ref, t0, i_init, s0):
        Ts, N = ref.sampling_rate, ref.cont_horizon
        Vdc = inverter.V_dc  # TO BE FIXED FOR PMSM
        #w = 2.0 * math.pi * load.f_backemf   # TO BE FIXED FOR PMSM
        ia0, ib0, ic0 = map(float, i_init)
        ia_ref, ib_ref, ic_ref = ref.generateThreePhaseRefs(t0)

        # Full brute force up to N<=3 to guarantee performance ↑ with horizon.
        if N <= 3:
            bestJ = float('inf'); best_seq = None
            for seq in itertools.product(self._abc_set, repeat=N):
                ia, ib, ic = ia0, ib0, ic0
                Ia, Ib, Ic = [], [], []
                for k,(sa,sb,sc) in enumerate(seq):
                    # pole voltages
                    vaN = Vdc*(2*sa-1)/2.0
                    vbN = Vdc*(2*sb-1)/2.0
                    vcN = Vdc*(2*sc-1)/2.0
                    # zero-seq removal
                    v0 = (vaN + vbN + vcN)/3.0
                    va, vb, vc = vaN - v0, vbN - v0, vcN - v0
                    # circuit step
                    ia = load._electrical_step_3phase(va, vb, vc, ia, ib, ic, Ts)[0]
                    ib = load._electrical_step_3phase(va, vb, vc, ia, ib, ic, Ts)[1]
                    ic = load._electrical_step_3phase(va, vb, vc, ia, ib, ic, Ts)[2]
                    Ia.append(ia); Ib.append(ib); Ic.append(ic)
                J = self._cost_three(Ia, Ib, Ic, ia_ref, ib_ref, ic_ref, seq, s0)
                if J < bestJ:
                    bestJ, best_seq = J, seq
            return list(best_seq), bestJ

        # If N>3 and Gurobi is enabled/available, try MIQP
        if self.use_gurobi and self._has_grb:
            try:
                return self._miqp_three(inverter, load, ref, t0, i_init, s0)
            except Exception:
                pass

        # Else, beam search to maintain forward-looking optimization
        return self._beam_three(inverter, load, ref, t0, i_init, s0, beam_width=64)

    # ----------------------- beam search (3φ, N>3) --------------------
    def _beam_three(self, inverter, load, ref, t0, i_init, s0, beam_width=64):
        Ts, N = ref.sampling_rate, ref.cont_horizon
        Vdc  = inverter.V_dc  
        ia_ref, ib_ref, ic_ref = ref.generateThreePhaseRefs(t0)

        beams = [ ([], i_init, 0.0) ]  # (seq, (ia,ib,ic), J_so_far)
        for k in range(N):
            new = []
            for seq, (ia,ib,ic), Jacc in beams:
                for abc in self._abc_set:
                    sa,sb,sc = abc
                    vaN=Vdc*(2*sa-1)/2.0; vbN=Vdc*(2*sb-1)/2.0; vcN=Vdc*(2*sc-1)/2.0
                    v0=(vaN+vbN+vcN)/3.0; va=vaN-v0; vb=vbN-v0; vc=vcN-v0
                    # circuit step
                    ia2 = load._electrical_step_3phase(va, vb, vc, ia, ib, ic, Ts)[0]
                    ib2 = load._electrical_step_3phase(va, vb, vc, ia, ib, ic, Ts)[1]
                    ic2 = load._electrical_step_3phase(va, vb, vc, ia, ib, ic, Ts)[2]
                    # incremental αβ tracking at step k (using ref[k])
                    iα,iβ = clarke_power_invariant([ia2],[ib2],[ic2])
                    iαr,iβr= clarke_power_invariant([ia_ref[k]],[ib_ref[k]],[ic_ref[k]])
                    track = float((iα-iαr)**2 + (iβ-iβr)**2)
                    if k==0 and s0:
                        psa,psb,psc=s0[0]
                    else:
                        psa,psb,psc= (seq[-1] if seq else (sa,sb,sc))
                    sw = (sa-psa)**2+(sb-psb)**2+(sc-psc)**2
                    new.append((seq+[abc], (ia2,ib2,ic2), Jacc + track + self.lambda_switch*sw))
            # prune
            new.sort(key=lambda t:t[2])
            beams = new[:beam_width]
        best = min(beams, key=lambda t:t[2])
        return best[0], float(best[2])

    # ------------------------ Gurobi MIQP ----------------------------
    def _miqp_three(self, inverter, load, ref, t0, i_init, s0):
        import gurobipy as gp
        from gurobipy import GRB

        Ts, N = ref.sampling_rate, ref.cont_horizon
        Vdc, R, L = inverter.V_dc, load.R, load.L   # TO BE FIXED FOR PMSM
        w = 2.0 * math.pi * load.f_backemf  # TO BE FIXED FOR PMSM
        ia0, ib0, ic0 = map(float, i_init)
        ia_ref, ib_ref, ic_ref = ref.generateThreePhaseRefs(t0)

        # Precompute back-emfs
        Ea = np.array([load.V_backemf * math.sin(w*(t0 + k*Ts)) for k in range(N)], dtype=float)    # TO BE FIXED FOR PMSM
        Eb = np.array([load.V_backemf * math.sin(w*(t0 + k*Ts) - 2.0*math.pi/3.0) for k in range(N)], dtype=float)  # TO BE FIXED FOR PMSM
        Ec = np.array([load.V_backemf * math.sin(w*(t0 + k*Ts) + 2.0*math.pi/3.0) for k in range(N)], dtype=float)  # TO BE FIXED FOR PMSM

        m = gp.Model("fsmpc_three")
        m.Params.OutputFlag=0; m.Params.TimeLimit=self.timelimit; m.Params.MIPGap=self.mipgap; m.Params.Threads=self.threads
        m.Params.NumericFocus=3; m.Params.ScaleFlag=2

        sa = m.addVars(N, vtype=GRB.BINARY, name="sa")
        sb = m.addVars(N, vtype=GRB.BINARY, name="sb")
        sc = m.addVars(N, vtype=GRB.BINARY, name="sc")
        ia = m.addVars(N+1, lb=-GRB.INFINITY, name="ia")
        ib = m.addVars(N+1, lb=-GRB.INFINITY, name="ib")
        ic = m.addVars(N+1, lb=-GRB.INFINITY, name="ic")

        m.addConstr(ia[0] == ia0); m.addConstr(ib[0] == ib0); m.addConstr(ic[0] == ic0)

        for k in range(N):
            vaN = (Vdc/2.0) * (2*sa[k]-1)
            vbN = (Vdc/2.0) * (2*sb[k]-1)
            vcN = (Vdc/2.0) * (2*sc[k]-1)
            v0  = (vaN + vbN + vcN)/3.0
            va, vb, vc = vaN - v0, vbN - v0, vcN - v0

            m.addConstr(ia[k+1] == ia[k] + Ts*((va - Ea[k] - R*ia[k])/L), name=f"da_{k}")   # TO BE FIXED FOR PMSM
            m.addConstr(ib[k+1] == ib[k] + Ts*((vb - Eb[k] - R*ib[k])/L), name=f"db_{k}")   # TO BE FIXED FOR PMSM
            m.addConstr(ic[k+1] == ic[k] + Ts*((vc - Ec[k] - R*ic[k])/L), name=f"dc_{k}")   # TO BE FIXED FOR PMSM

        # αβ tracking cost (quadratic)
        obj = gp.QuadExpr()
        for k in range(N):
            # predicted (k+1) vs ref (k)
            iα_pred, iβ_pred = clarke_power_invariant(
                [ia[k+1]], [ib[k+1]], [ic[k+1]]
            )
            iα_ref,  iβ_ref  = clarke_power_invariant(
                [float(ia_ref[k])], [float(ib_ref[k])], [float(ic_ref[k])]
            )
            obj += (iα_pred - iα_ref)*(iα_pred - iα_ref) + (iβ_pred - iβ_ref)*(iβ_pred - iβ_ref)

            # Δs terms
            if k==0 and s0:
                psa,psb,psc = s0[0]
                obj += self.lambda_switch * ((sa[k]-int(psa))*(sa[k]-int(psa)) +
                                             (sb[k]-int(psb))*(sb[k]-int(psb)) +
                                             (sc[k]-int(psc))*(sc[k]-int(psc)))
            else:
                obj += self.lambda_switch * ((sa[k]-sa[k-1])*(sa[k]-sa[k-1]) +
                                             (sb[k]-sb[k-1])*(sb[k]-sb[k-1]) +
                                             (sc[k]-sc[k-1])*(sc[k]-sc[k-1]))
        m.setObjective(obj, GRB.MINIMIZE)
        m.optimize()

        if m.SolCount>0:
            seq=[(int(round(sa[k].X)), int(round(sb[k].X)), int(round(sc[k].X))) for k in range(N)]
            return seq, float(m.ObjVal)
        raise RuntimeError("No MIQP solution")

    # ----------------------- safe-mode PI ----------------------------
    def _safe_pi(self, ref_now, i_now):
        e = ref_now - i_now
        self._int_err_a += e*1e-4
        u = self.safe_Kp*e + self.safe_Ki*self._int_err_a
        return 1 if u >= 0 else 0

    def _safe_pi_three(self, ref_now_abc, i_now_abc):
        s = []
        for idx,(r,i) in enumerate(zip(ref_now_abc, i_now_abc)):
            if idx==0: self._int_err_a += (r-i)*1e-4; u = self.safe_Kp*(r-i) + self.safe_Ki*self._int_err_a
            if idx==1: self._int_err_b += (r-i)*1e-4; u = self.safe_Kp*(r-i) + self.safe_Ki*self._int_err_b
            if idx==2: self._int_err_c += (r-i)*1e-4; u = self.safe_Kp*(r-i) + self.safe_Ki*self._int_err_c
            s.append(1 if u>=0 else 0)
        return tuple(s)

    # -------------------------- public API ---------------------------
    def solveMPC(self, inverter, load, currentReference, current_time, i_init, s0=None):
        Ts, N = currentReference.sampling_rate, self.cont_horizon

        if s0 is None: s0 = [(True,True,True)]*N
        # 1) brute force (N<=3) else beam or Gurobi
        try:
            seq, J = self._bf_three(inverter, load, currentReference, current_time, i_init, s0)
            self._last_action = seq[0] if isinstance(seq[0], int) else seq[0]
            return seq, J
        except Exception:
            pass
        # 2) optional Gurobi (already attempted inside _bf_three when N>3)
        if self.use_gurobi and self._has_grb:
            try:
                seq, J = self._miqp_three(inverter, load, currentReference, current_time, i_init, s0)
                self._last_action = seq[0]
                return seq, J
            except Exception:
                pass
        # 3) PI fallback
        ia_ref, ib_ref, ic_ref = currentReference.generateThreePhaseRefs(current_time)
        s = self._safe_pi_three((ia_ref[0], ib_ref[0], ic_ref[0]), i_init)
        self._last_action = s
        return [s]*N, 1e9