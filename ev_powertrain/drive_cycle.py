
import csv

def load_csv_cycle(path):
    """
    CSV with columns: t (s), v_mps (m/s)
    Returns two lists (t, v).
    """
    t=[]; v=[]
    with open(path, 'r', newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            t.append(float(row['t']))
            v.append(float(row['v_mps']))
    return t, v

def sample_udds_short():
    # 0..300 s snippet with starts, stops, and cruises
    t=[]; v=[]
    import math
    dt=0.1; T=300.0; tcur=0.0
    while tcur <= T+1e-9:
        # piecewise: accel to 15 m/s, cruise, decel, stop repeat
        mod = tcur % 60.0
        if mod < 10.0:
            vref = 1.5*mod  # accel to 15 m/s in 10 s
        elif mod < 35.0:
            vref = 15.0
        elif mod < 45.0:
            vref = max(0.0, 15.0 - 1.5*(mod-35.0))  # decel to 0 in 10 s
        else:
            vref = 0.0
        t.append(round(tcur,3)); v.append(vref)
        tcur += dt
    return t, v
