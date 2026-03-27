# === two_level_inverter_fsmpc/scenario_executor/scenario_exc.py ===
import math
import numpy as np

def sim_executor(
    load,
    inverter,
    mpc,
    currentReference,
    s0,
    t_0=0.0,
    i_a_0=0.0,
    sampling_rate=1e-4,
    sim_time=0.1,
    i_b0=0.0,
    i_c0=0.0,
    slower_by=10,
    # --- FOC/PMSM optional ---
    ang_speed_controller=None,
    mtpa=None,
    # drive cycle (optional)
    t_cycle=None,
    v_cycle=None,
    idx_cycle=0,
    # Vehicle speed control optional blocks:
    driver=None,
    vehicle=None,
    ):
    """
    Simulation loop owner. Keeps the legacy API but adds optional PMSM FOC path.

    Returns
    -------
    (s_traj, (i_a_traj, i_b_traj, i_c_traj),
    (i_ref_a_traj, i_ref_b_traj, i_ref_c_traj), t_sim, cost_func_val)
    """
    steps = max(0, int(round((sim_time - t_0) / sampling_rate)))
    current_time = t_0

    # initialize currents
    i_a = float(i_a_0)
    i_b = float(i_b0)
    i_c = float(i_c0)

    ## Logs
    # Time trajectory
    t_sim = []
    # Current Controller
    s_traj = []
    cost_func_val = []
    v_a_traj, v_b_traj, v_c_traj = [], [], []
    i_a_traj, i_ref_a_traj = [], []
    i_b_traj, i_ref_b_traj = [], []
    i_c_traj, i_ref_c_traj = [], []
    i_d_traj, i_q_traj = [], []
    i_q_cmd_traj = []
    # Angular Speed Controller 
    omega_m_traj = []
    omega_ref_traj = []
    T_e_traj = []
    # Vehicle Speed Controller
    v_ref_traj = []
    v_veh_traj = []
    T_L_traj = []

    Ts = sampling_rate

    for i in range(steps):
        if i % slower_by == 0:
            # driver cycle index
            while idx_cycle+1 < len(t_cycle) and t_cycle[idx_cycle+1] <= current_time:
                idx_cycle += 1
            v_ref = v_cycle[idx_cycle]
            # --- Vehicle speed control (optional) ---
            if driver is not None and vehicle is not None:
                # current vehicle speed and motor speed
                v_meas = vehicle.v
                # driver converts v_ref->omega_ref
                omega_ref, _ = driver.step(v_ref, v_meas, vehicle)
                
            
            # load.T_L = vehicle.forces()[3]*vehicle.Rw/vehicle.G  # load torque
            # omega_meas = load.get_omega_m() # v_meas / vehicle.Rw * vehicle.G  # approximate motor speed from vehicle speed
            # ---- Vehicle Speed PI -> (optional) MTPA -> dq refs -> αβ refs ----
            i_q_cmd = omega_ref

            if mtpa is not None:
                i_d_ref, i_q_ref = mtpa.compute_from_iq(i_q_cmd)
            else:
                i_d_ref, i_q_ref = 0.0, i_q_cmd/vehicle.Rw

            theta_e = load.get_theta_e()
            i_alpha_ref, i_beta_ref = currentReference.set_dq_refs(i_d_ref, i_q_ref, theta_e)

            # Provide ABC reference (length-1) compatible with solver API.
            i_a_ref = i_alpha_ref
            i_b_ref = -0.5*i_alpha_ref + (math.sqrt(3)/2)*i_beta_ref
            i_c_ref = -0.5*i_alpha_ref - (math.sqrt(3)/2)*i_beta_ref

        # Build a horizon-length constant ref (repeat current target across horizon)
        N = mpc.cont_horizon
        ia_seq = np.array([i_a_ref]*N, dtype=float)
        ib_seq = np.array([i_b_ref]*N, dtype=float)
        ic_seq = np.array([i_c_ref]*N, dtype=float)

        # Adapter: make solver see these refs via its usual method
        def _gen3(_t):
            return ia_seq, ib_seq, ic_seq
        currentReference.generateThreePhaseRefs = _gen3

        # --- solve FS-MPC for the full horizon (3-phase) ---
        load.T_L = vehicle.forces()[3]*vehicle.Rw/vehicle.G  # load torque
        s_seq, J = mpc.solveMPC(inverter, load, currentReference, current_time,
                                i_init=(i_a, i_b, i_c), s0=s0)

        sa, sb, sc = map(int, s_seq[0])

        # Inverter pole voltages
        vaN = inverter.V_dc * (2*sa - 1) / 2.0
        vbN = inverter.V_dc * (2*sb - 1) / 2.0
        vcN = inverter.V_dc * (2*sc - 1) / 2.0
        # Zero-sequence removal for three-wire PMSM
        v0 = (vaN + vbN + vcN) / 3.0
        va, vb, vc = vaN - v0, vbN - v0, vcN - v0
        v_a_traj.append(float(va)); v_b_traj.append(float(vb)); v_c_traj.append(float(vc))

        ## Log PMSM trajectories (electrical) ##
        (i_a, i_b, i_c), (i_d, i_q), T_e, omega_m = load.step(va, vb, vc, i_a, i_b, i_c, Ts)
        # abc refs (scalars at this sample)
        i_ref_a_traj.append(float(i_a_ref))
        i_ref_b_traj.append(float(i_b_ref))
        i_ref_c_traj.append(float(i_c_ref))
        # dq currents
        i_d_traj.append(float(i_d))
        i_q_traj.append(float(i_q))
        i_q_cmd_traj.append(float(i_q_cmd))
        # abc motor currents
        i_a_traj.append(i_a); i_b_traj.append(i_b); i_c_traj.append(i_c)

        ## Log PMSM trajectories (mechanical) ##
        omega_m = load.get_omega_m()
        omega_ref_traj.append(omega_ref/vehicle.Rw)
        omega_m_traj.append(float(omega_m))
        T_e_traj.append(float(T_e))

        # Log vehicle trajectories
        v_ref_traj.append(v_ref)
        v_veh_traj.append(vehicle.v)
        T_L_traj.append(load.T_L)

        # Log applied switch states & cost function and simulation time
        s_traj.append((sa, sb, sc))
        cost_func_val.append(float(J))
        t_sim.append(current_time)

        # warm-start: repeat the applied action over the horizon
        psa, psb, psc = s_traj[-1]
        s0 = [(bool(psa), bool(psb), bool(psc))] * mpc.cont_horizon

        # Couple motor torque to vehicle
        vehicle.v_next(float(T_e), Ts)

        # Increment the next simulation time step
        current_time += Ts

    return (
        s_traj,
        (i_a_traj, i_b_traj, i_c_traj),
        (i_ref_a_traj, i_ref_b_traj, i_ref_c_traj),
        (v_a_traj, v_b_traj, v_c_traj),
        t_sim,
        cost_func_val,
        (omega_ref_traj, omega_m_traj),
        T_e_traj,
        (i_d_traj,
        i_q_traj, i_q_cmd_traj),
        (v_ref_traj, v_veh_traj),
        T_L_traj
    )