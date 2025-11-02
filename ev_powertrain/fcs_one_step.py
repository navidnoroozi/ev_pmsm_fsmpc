
import math

ABC_SET = [(0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)]

def dq_to_alpha_beta(i_d, i_q, theta):
    c,s = math.cos(theta), math.sin(theta)
    i_alpha = c*i_d - s*i_q
    i_beta  = s*i_d + c*i_q
    return i_alpha, i_beta

def three_from_alpha_beta(i_alpha, i_beta):
    ia = i_alpha
    ib = -0.5*i_alpha + (math.sqrt(3)/2.0)*i_beta
    ic = -0.5*i_alpha - (math.sqrt(3)/2.0)*i_beta
    return ia, ib, ic

def clarke_power_invariant(ia, ib, ic):
    SQ23 = (2.0/3.0)**0.5
    SQ3_2 = (3.0**0.5)/2.0
    i_alpha = SQ23 * (ia - 0.5*ib - 0.5*ic)
    i_beta  = SQ23 * (SQ3_2*(ib - ic))
    return i_alpha, i_beta

def select_switch_one_step(load, Vdc, ia, ib, ic, i_d_ref, i_q_ref, theta_e, Ts, s_prev=(1,1,1), lambda_sw=0.05):
    # build 3-phase references from dq
    ia_r, ib_r, ic_r = three_from_alpha_beta(*dq_to_alpha_beta(i_d_ref, i_q_ref, theta_e))
    bestJ=1e99; best=(1,1,1)
    for sa,sb,sc in ABC_SET:
        # pole voltages and zero-seq removal
        vaN = Vdc*(2*sa-1)/2.0; vbN = Vdc*(2*sb-1)/2.0; vcN = Vdc*(2*sc-1)/2.0
        v0 = (vaN+vbN+vcN)/3.0
        va, vb, vc = vaN-v0, vbN-v0, vcN-v0
        # predict one step using user's electrical substep (no mech coupling here)
        ia_p, ib_p, ic_p, *_ = load._electrical_step_3phase(va, vb, vc, ia, ib, ic, Ts)
        # αβ cost
        iα,iβ = clarke_power_invariant(ia_p, ib_p, ic_p)
        iαr,iβr= clarke_power_invariant(ia_r, ib_r, ic_r)
        track = (iα-iαr)**2 + (iβ-iβr)**2
        sw = (sa-s_prev[0])**2 + (sb-s_prev[1])**2 + (sc-s_prev[2])**2
        J = track + lambda_sw*sw
        if J < bestJ:
            bestJ, best = J, (sa,sb,sc)
    return best, bestJ
