# For plotting
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

# Plotting currents and switching signals
def plot_simulation_results(t_sim, 
                            i_a_traj, i_b_traj, i_c_traj, i_ref_a_traj, i_ref_b_traj, i_ref_c_traj,
                            v_a_traj, v_b_traj, v_c_traj, 
                            s_a, s_b, s_c, 
                            omega_ref_traj, omega_m_traj, 
                            T_e_traj, T_L_traj, 
                            i_d_traj, i_q_traj, i_q_cmd_traj, 
                            cost_func_val, 
                            v_ref_traj, v_veh_traj, 
                            show_plots=True):
        fig_iabc = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=("Phase Currents", "Current References"))
        fig_iabc.add_trace(go.Scatter(x=t_sim, y=i_a_traj, mode='lines', name='i_a'), row=1, col=1)
        fig_iabc.add_trace(go.Scatter(x=t_sim, y=i_b_traj, mode='lines', name='i_b'), row=1, col=1)
        fig_iabc.add_trace(go.Scatter(x=t_sim, y=i_c_traj, mode='lines', name='i_c'), row=1, col=1)
        fig_iabc.add_trace(go.Scatter(x=t_sim, y=i_ref_a_traj, mode='lines', name='i_ref_a'), row=2, col=1)
        fig_iabc.add_trace(go.Scatter(x=t_sim, y=i_ref_b_traj, mode='lines', name='i_ref_b'), row=2, col=1)
        fig_iabc.add_trace(go.Scatter(x=t_sim, y=i_ref_c_traj, mode='lines', name='i_ref_c'), row=2, col=1)
        fig_iabc.update_layout(title_text="Phase Currents and References", xaxis_title="Time (s)", yaxis_title="Current (A)", height=800)

        # if show_plots: fig_iabc.show()
        fig_volt_sw = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                     subplot_titles=("Phase Voltages", "Inverter Switching Signals"))
        fig_volt_sw.add_trace(go.Scatter(x=t_sim, y=v_a_traj, mode='lines', name='v_a', line=dict(shape="hv")), row=1, col=1)
        fig_volt_sw.add_trace(go.Scatter(x=t_sim, y=v_b_traj, mode='lines', name='v_b', line=dict(shape="hv")), row=1, col=1)
        fig_volt_sw.add_trace(go.Scatter(x=t_sim, y=v_c_traj, mode='lines', name='v_c', line=dict(shape="hv")), row=1, col=1)
        fig_volt_sw.add_trace(go.Scatter(x=t_sim, y=s_a, mode='lines', name='s_a', line=dict(shape="hv")), row=2, col=1)
        fig_volt_sw.add_trace(go.Scatter(x=t_sim, y=s_b, mode='lines', name='s_b', line=dict(shape="hv")), row=2, col=1)
        fig_volt_sw.add_trace(go.Scatter(x=t_sim, y=s_c, mode='lines', name='s_c', line=dict(shape="hv")), row=2, col=1)
        fig_volt_sw.update_layout(
            title_text="Phase Voltages and Inverter Switching Signals",
            xaxis_title="Time (s)",
            yaxis_title="Voltage (V)",
            yaxis2_title="Switching Signal",
            height=800
        )
        if show_plots:
            fig_volt_sw.show()

        fig_speed = go.Figure()
        fig_speed.add_trace(go.Scatter(x=t_sim, y=omega_m_traj, mode='lines', name='Mechanical Speed (rad/s)'))
        fig_speed.add_trace(go.Scatter(x=t_sim, y=omega_ref_traj, mode='lines', name='Reference Speed (rad/s)', line=dict(dash='dash')))
        fig_speed.update_layout(title="Motor Angular Speed over Time", xaxis_title="Time (s)", yaxis_title="Omega_m (rad/s)")
        if show_plots:
            fig_speed.show()

        fig_theta = go.Figure()
        fig_theta.add_trace(go.Scatter(x=t_sim, y=T_e_traj, mode='lines', name='Electrical Torque (Nm)'))
        fig_theta.add_trace(go.Scatter(x=t_sim, y=T_L_traj, mode='lines', name='Load Torque (Nm)', line=dict(dash='dash')))
        fig_theta.update_layout(title="Torque over Time", xaxis_title="Time (s)", yaxis_title="T (Nm)")
        if show_plots:
            fig_theta.show()

        fig_dq = go.Figure()
        fig_dq.add_trace(go.Scatter(x=t_sim, y=i_d_traj, mode='lines', name='i_d'))
        fig_dq.add_trace(go.Scatter(x=t_sim, y=i_q_traj, mode='lines', name='i_q'))
        fig_dq.add_trace(go.Scatter(x=t_sim, y=i_q_cmd_traj, mode='lines', name='i_q_cmd'))
        fig_dq.update_layout(title="dq Currents over Time", xaxis_title="Time (s)", yaxis_title="Current (A)")
        if show_plots:
            fig_dq.show()

        fid_c_func = go.Figure()
        fid_c_func.add_trace(go.Scatter(x=t_sim, y=cost_func_val, mode='lines', name='Cost Function Value'))
        fid_c_func.update_layout(title="Cost Function Value over Time", xaxis_title="Time (s)", yaxis_title="Cost Function Value")
        if show_plots:
            fid_c_func.show()

        fig_veh_speed = go.Figure()
        fig_veh_speed.add_trace(go.Scatter(x=t_sim, y=v_veh_traj, mode='lines', name='Vehicle Speed (m/s)'))
        fig_veh_speed.add_trace(go.Scatter(x=t_sim, y=v_ref_traj, mode='lines', name='Reference Vehicle Speed (m/s)', line=dict(dash='dash')))
        fig_veh_speed.update_layout(title="Vehicle Speed over Time", xaxis_title="Time (s)", yaxis_title="Vehicle Speed (m/s)")
        if show_plots:
            fig_veh_speed.show()