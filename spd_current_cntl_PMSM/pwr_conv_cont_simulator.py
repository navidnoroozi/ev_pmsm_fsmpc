import sys, math, csv
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent/"ev_powertrain"))

# EV blocks
from vehicle import Vehicle1D
from driver import DriverPI
from fcs_one_step import select_switch_one_step
from drive_cycle import sample_udds_short

# Core blocks
from inverter.inverter_behave import Inverter
from mpc_contr.mpc_contr_calc import MPCSSolver
from scenario_executor.scenario_exc import sim_executor

# PMSM FOC blocks
from load.load_dyn_cal import PMSMLoad
from current_reference.current_ref_gen import CurrentReference
from speed_ctrl.speed_controller import SpeedController
from opt_ref.mtpa import MTPA

# Plotting
import plot.plot_results as plot_results

# Performance reporting
import report.performance_report as perf_report

L_s = 8.2e-3
# --- simulation setup ---
Ts_e = 30e-6      # electrical sample (inner loop)
slower_by = 200  # run driver+speed controller every slow*Ts_e (6 ms)
sim_time = 1e5*Ts_e # 300.0  # seconds (matches sample cycle length)

def simPowertrainControlSyst(
    # --- voltage requests ---
    V_rms_req=230.0,
    show_plots=True,

    # --- control horizon & sim setup ---
    cont_horizon=2,
    t_0=0.0,
    i_a_0=0.0,
    sampling_rate=Ts_e,
    slower_by = slower_by,
    sim_time=sim_time,

    # --- plant params (PMSM defaults; remain configurable) ---
    R_s=0.3, pole_pairs=3,
    L_d=L_s, L_q=L_s,
    psi_f=0.125, J=4e-3, B=1e-3,

    # --- inverter & grid/reference params ---
    V_dc=400.0,       # if None, derived from V_rms_req
    use_mtpa=False,   # << enable MTPA between speed PI and current refs
    use_gurobi=False,
    timelimit=0.002,
    mipgap=0.05,
    lambda_switch=0.1,
    threads=1,
):
    """
    Top-level orchestrator that builds the objects and defers the simulation loop
    to scenario_executor.sim_executor. Supports:
      - Three-phase PMSM FOC + FS-MPC
    """

    # --- Vehicle & Drivetrain ---
    veh = Vehicle1D(m=1700.0, Cr=0.010, rho=1.2, CdA=0.70, Rw=0.31, G=9.0, eta=0.95)

    # --- Drive Cycle ---
    t_cycle, v_cycle = sample_udds_short()

    # --- Inverter DC link ---
    inverter = Inverter(V_dc=V_dc)
    
    # --- PMSM & Current References ---
    load = PMSMLoad(
        R_s=R_s, L_d=L_d, L_q=L_q, psi_f=psi_f, pole_pairs=pole_pairs,
        J=J, B=B,
    )
    currentReference = CurrentReference(sampling_rate, cont_horizon)
    
    ## --- Controllers ---
    # Outer-most loop: Speed PI controller (30 Hz loop on outer-most driver is too fast; use gentle gains here)
    driver = DriverPI(Kp=0.6, Ki=0.2, Ts=slower_by*sampling_rate)

    # Outer loop angular speed PI controller + (optional) MTPA
    ang_speed_ctrl = SpeedController(Kp=2.41, Ki=253.0, Ts=slower_by*sampling_rate, i_q_ref_max=250.0)
    mtpa = MTPA(L_d=L_d, L_q=L_q, psi_f=psi_f, pole_pairs=pole_pairs) if use_mtpa else None

    # Inner-most loop: MPC for current control
    mpc = MPCSSolver(
        cont_horizon=cont_horizon,
        lambda_switch=lambda_switch,
        use_gurobi=use_gurobi,
        timelimit=timelimit,
        mipgap=mipgap,
        threads=threads,
    )

    # --- initial switching warm-start ---
    s0 = [(True, True, True)] * cont_horizon

    # --- run scenario executor (single loop owner) ---
    results = sim_executor(
        load=load,
        inverter=inverter,
        mpc=mpc,
        currentReference=currentReference,
        s0=s0,
        t_0=t_0,
        i_a_0=i_a_0,
        sampling_rate=sampling_rate,
        sim_time=sim_time,
        slower_by=slower_by,
        # --- FOC/PMSM optional ---
        ang_speed_controller=ang_speed_ctrl,
        mtpa=None,
        # drive cycle (optional)
        t_cycle=t_cycle,
        v_cycle=v_cycle,
        idx_cycle=0,
        # Vehicle speed control optional blocks:
        driver=driver,
        vehicle=veh)

    # Unpack and make plots similar to your earlier style
    s_traj, (i_a_traj, i_b_traj, i_c_traj), (i_ref_a_traj, 
    i_ref_b_traj, i_ref_c_traj), (v_a_traj, v_b_traj, 
    v_c_traj), t_sim, cost_func_val, (omega_ref_traj, omega_m_traj), T_e_traj, (i_d_traj, 
    i_q_traj, i_q_cmd_traj), (v_ref_traj, v_veh_traj), T_L_traj = results
    
    # Extract switching signals for each phase
    s_a = [s[0] for s in s_traj]
    s_b = [s[1] for s in s_traj]
    s_c = [s[2] for s in s_traj]

    # Plot results
    plot_results.plot_simulation_results(
        t_sim,
        i_a_traj, i_b_traj, i_c_traj,
        i_ref_a_traj, i_ref_b_traj, i_ref_c_traj,
        v_a_traj, v_b_traj, v_c_traj,
        s_a, s_b, s_c,
        omega_ref_traj,
        omega_m_traj,
        T_e_traj,
        T_L_traj,
        i_d_traj, i_q_traj, i_q_cmd_traj,
        cost_func_val,
        v_ref_traj, v_veh_traj,
        show_plots=show_plots
    )

    # --- PERFORMANCE REPORT GENERATION ---
    # Prepare essential data for performance report
    f_fund = 50.0  # fundamental frequency for harmonic analysis
    fs = 1 / sampling_rate
    max_harm = 20    # maximum harmonic order for analysis
    omega_m_traj = np.array(omega_m_traj)
    torque_traj = np.array(T_e_traj)
    vabc_traj = np.array([v_a_traj, v_b_traj, v_c_traj])
    iabc_traj = np.array([i_a_traj, i_b_traj, i_c_traj])
    P_nominal_motor = 4e3  # Example nominal power in Watts
    I_rated = P_nominal_motor / (math.sqrt(3) * V_rms_req)  # Rated current for TDD calculation
    # Generate performance report
    report_df, harmonic_data = perf_report.create_performance_report(
    t_sim,
    omega_ref_traj,
    omega_m_traj,
    torque_traj,
    vabc_traj,
    iabc_traj,
    fs,
    f_fund,
    I_rated,
    max_harm,
    isc_il=100,
    simulation_name="Test Run #1"
    )

    perf_report.print_report(report_df)      # → prints clean summary to terminal
    perf_report.save_report_csv(report_df)   # → stores results to CSV

    return results
if __name__ == "__main__":
    simPowertrainControlSyst(show_plots=True)