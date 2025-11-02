
import math

def clarke_trans_calc_in_balanced(ia_seq, ib_seq, ic_seq):
    """Balanced Clarke transform (power-invariant).
    Inputs are iterables of equal length; returns (i_alpha_seq, i_beta_seq) lists.
    i_alpha = sqrt(2/3)*(ia - 0.5*ib - 0.5*ic)
    i_beta  = sqrt(2/3)*(sqrt(3)/2)*(ib - ic)
    """
    ia = list(ia_seq); ib = list(ib_seq); ic = list(ic_seq)
    assert len(ia)==len(ib)==len(ic), "abc sequences must have same length"
    k = math.sqrt(2.0/3.0); k2 = math.sqrt(3.0)/2.0
    a=[]; b=[]
    for i in range(len(ia)):
        a.append(k*(ia[i] - 0.5*ib[i] - 0.5*ic[i]))
        b.append(k*(k2*(ib[i] - ic[i])))
    return a,b
