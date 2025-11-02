# === performance_report.py ===
import numpy as np
import pandas as pd
# Performance metrics
import performance_eval_calc.performance_metrics as perf_metrics
import performance_eval_calc.performance_ieee519 as ieee519_metrics


def create_performance_report(
        t, omega_ref, omega_meas,
        torque_traj,
        v_abc, i_abc,
        fs, f_fund, I_rated, max_harm,
        isc_il=100,
        simulation_name="Simulation",
        do_plot=True,
    ):
    """
    Generate a structured performance report for console / CSV usage.
    """

    # ---- Speed metrics ----
    speed_results = perf_metrics.step_metrics(t, omega_meas, omega_ref)

    # ---- Torque ripple ----
    torque_rip = perf_metrics.torque_ripple(torque_traj)

    # ---- Efficiency & power factor ----
    pf, eff = perf_metrics.efficiency_powerfactor(v_abc, i_abc)

    # Compute TDD for phase A current
    i_a = i_abc[0]
    perf_metrics.compute_tdd(i_a, fs, I_rated, plot_fft=True)

    # ---- IEEE 519 harmonic analysis ----
    harmonic_results = ieee519_metrics.analyze_ieee519_3ph(
        i_abc[0], i_abc[1], i_abc[2],
        fs=fs, f_fund=f_fund,
        I_rated=I_rated,
        isc_il=isc_il,
        max_harm=max_harm,
        do_plot=do_plot,
        title="IEEE 519 Harmonic Analysis (3-Phase)"
    )

    # Combine summary results
    report = {
        "Simulation Name": simulation_name,
        "Speed Rise Time (s)": speed_results["rise_time"],
        "Speed Overshoot (%)": speed_results["overshoot"],
        "Speed Settling Time (s)": speed_results["settling_time"],
        "Speed Steady-State Error": speed_results["steady_state_error"],
        "Torque Ripple (%)": torque_rip,
        "Efficiency (%)": eff,
        "Power Factor": pf,
        "THD Average (%)": harmonic_results["THD_avg%"],
        "TDD Average (%)": harmonic_results["TDD_avg%"],
        "IEEE 519 Compliance": "PASS" if harmonic_results["TDD_avg%"] <= 5 else "FAIL",
        "ISC/IL": isc_il
    }

    return pd.DataFrame([report]), harmonic_results


def print_report(report_df):
    """
    Pretty-print report to terminal in clean table-like format.
    """
    print("\n=== PERFORMANCE REPORT ===")
    for col, val in report_df.iloc[0].items():
        print(f"{col:30s}: {val}")
    print("="*32 + "\n")


def save_report_csv(report_df, filename="performance_report.csv"):
    """
    Save report to .csv file.
    """
    report_df.to_csv(filename, index=False)
    print(f"[✓] Report saved to {filename}")


    # # ---------- STEP RESPONSE METRICS FOR MECHANICAL SPEED ----------
    # step_metrics = perf_metrics.step_metrics(t_sim, omega_m_traj, omega_ref)
    # print("Step Response Metrics for Mechanical Speed:")
    # for k, v in step_metrics.items():
    #     print(f"  {k}: {v:.4f}")
    
    # # ---------- TORQUE RIPPLE ----------
    # torque_ripple = perf_metrics.torque_ripple(T_e_traj)
    # print(f"Torque Ripple: {torque_ripple:.2f}%")

    # # ---------- EFFICIENCY / PF ----------
    # efficiency, pf = perf_metrics.efficiency_powerfactor(v_traj, i_traj)
    # print(f"Efficiency: {efficiency:.2f}%, Power Factor: {pf:.2f}")