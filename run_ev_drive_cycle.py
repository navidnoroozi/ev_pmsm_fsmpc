
import sys, math, csv
from pathlib import Path
sys.path.append(str(Path(__file__).parent/"spd_current_cntl_PMSM"))
from load.load_dyn_cal import PMSMLoad
from inverter.inverter_behave import Inverter
from speed_ctrl.speed_controller import SpeedController
from ev_powertrain.vehicle import Vehicle1D
from ev_powertrain.driver import DriverPI
from ev_powertrain.fcs_one_step import select_switch_one_step
from ev_powertrain.drive_cycle import sample_udds_short

import matplotlib.pyplot as plt

def run():
    # --- simulation setup ---
    Ts_e = 30e-6      # electrical sample (inner loop)
    slow = 200        # run driver+speed controller every slow*Ts_e (6 ms)
    Ts_drv = Ts_e*slow
    sim_time = 300.0  # seconds (matches sample cycle length)

    # vehicle & drivetrain
    veh = Vehicle1D(m=1700.0, Cr=0.010, rho=1.2, CdA=0.70, Rw=0.31, G=9.0, eta=0.95)

    # PMSM & inverter
    load = PMSMLoad(R_s=0.3, L_d=0.008, L_q=0.008, psi_f=0.125, pole_pairs=3, J=4e-3, B=1e-3, T_L=veh.forces(0.0)[3]*veh.Rw/veh.G)  # load torque set to vehicle rolling resistance at 0 speed
    inv = Inverter(V_dc=400.0)
    Vdc = inv.V_dc
    load.T_L = veh.forces()[3]*veh.Rw/veh.G  # initial load torque

    # controllers
    # speed PI gains from earlier guidance (30 Hz loop on outer-most driver is too fast; use gentle gains here)
    driver = DriverPI(Kp=0.6, Ki=0.2, Ts=Ts_drv)
    speedPI = SpeedController(Kp=2.41, Ki=253.0, Ts=Ts_drv, i_q_ref_max=250.0)  # 30 Hz speed loop inside plant

    # drive cycle
    t_cycle, v_cycle = sample_udds_short()
    T_end = min(sim_time, t_cycle[-1])

    # states
    ia=0.0; ib=0.0; ic=0.0
    s_prev=(1,1,1)
    omega_ref = 0.0

    # logs
    log_t=[]; log_vref=[]; log_v=[]; log_om_ref=[]; log_om=[]; log_Te=[]; log_iq=[]; log_id=[]; log_s=[]

    k_driver = 0

    t=0.0
    idx_cycle = 0

    while t < T_end - 1e-12:
        # driver cycle index
        while idx_cycle+1 < len(t_cycle) and t_cycle[idx_cycle+1] <= t:
            idx_cycle += 1
        v_ref = v_cycle[idx_cycle]

        # update driver+speed loop at slower rate
        if k_driver % slow == 0:
            # current vehicle speed and motor speed
            v_meas = veh.v
            omega_meas = load.get_omega_m()
            # driver converts v_ref->omega_ref
            omega_ref, _ = driver.step(v_ref, v_meas, veh)
            # inner speed PI produces i_q_ref
            iq_ref, _ = speedPI.compute_iq_ref(omega_ref, omega_meas)
            id_ref = 0.0  # keep d=0; could connect MTPA later
        # choose switching for this electrical step (one-step ahead FCS)
        s, _ = select_switch_one_step(load, Vdc, ia, ib, ic, id_ref, iq_ref, load.get_theta_e(), Ts_e, s_prev=s_prev, lambda_sw=0.05)
        s_prev = s
        va_list, vb_list, vc_list = inv.generateOutputVoltage([s])
        va, vb, vc = va_list[0], vb_list[0], vc_list[0]

        # apply to load: returns next currents, torque, and motor speed
        (ia,ib,ic), (id_n, iq_n), Te, omega_m = load.step(va, vb, vc, ia, ib, ic, Ts_e)

        # couple motor torque to vehicle
        veh.v_next(Te, Ts_e)

        # log at slower rate to keep arrays small
        if k_driver % slow == 0:
            log_t.append(t)
            log_vref.append(v_ref)
            log_v.append(veh.v)
            log_om_ref.append(omega_ref)
            log_om.append(omega_m)
            log_Te.append(Te)
            log_iq.append(iq_n)
            log_id.append(id_n)
            log_s.append(s)

        k_driver += 1
        t += Ts_e

    # plots
    fig1 = plt.figure()
    plt.plot(log_t, log_vref, label="v_ref [m/s]")
    plt.plot(log_t, log_v, label="v [m/s]")
    plt.xlabel("t [s]"); plt.ylabel("Speed [m/s]"); plt.legend(); plt.title("Drive cycle tracking")

    fig2 = plt.figure()
    plt.plot(log_t, log_om_ref, label="omega_ref")
    plt.plot(log_t, log_om, label="omega")
    plt.xlabel("t [s]"); plt.ylabel("Motor speed [rad/s]"); plt.legend(); plt.title("Motor speed tracking")

    fig3 = plt.figure()
    plt.plot(log_t, log_Te, label="T_e [Nm]"); plt.xlabel("t [s]"); plt.ylabel("Torque [Nm]"); plt.title("Motor torque"); plt.legend()

    # save results
    outdir = Path(__file__).parent/"ev_results"
    outdir.mkdir(exist_ok=True)
    for i,fig in enumerate([fig1,fig2,fig3], start=1):
        fig.savefig(outdir/f"plot_{i}.png", dpi=150, bbox_inches="tight")

    # metrics
    import numpy as np
    vref = np.array(log_vref); v = np.array(log_v)
    rmse = float(np.sqrt(np.mean((v-vref)**2)))
    mae = float(np.mean(np.abs(v-vref)))
    with open(outdir/"metrics.txt","w") as f:
        f.write(f"Speed tracking RMSE (m/s): {rmse}\n")
        f.write(f"Speed tracking MAE  (m/s): {mae}\n")

    print("Saved plots to:", outdir)

if __name__ == "__main__":
    run()
