# === performance_metrics.py ===
import numpy as np
import matplotlib.pyplot as plt

# ---------- TIME-DOMAIN METRICS ----------
def step_metrics(t, y, y_ref):
    """
    Compute step response metrics: rise time, overshoot, settling time, steady-state error.
    Parameters:
    ----------
    t : np.ndarray
        Time vector (s)
    y : np.ndarray
        System response (V)
    y_ref : np.ndarray
        Reference value (V)
    Returns:
    -------
    dict : Step response metrics
    rise_time : float
        Time to reach 90% of steady-state value (s)
    overshoot : float
        Percent overshoot (%)
    settling_time : float
        Time to settle within ±2% of steady-state value (s)
    steady_state_error : float
        Difference between steady-state value and reference (V)
    """
    N_ss = min(20, len(y))  # Use last 20 samples or fewer if signal is short
    y_ss = np.mean(y[-N_ss:])
    tr_idx = np.where(y >= 0.9*y_ss)[0]
    # Rise time: time to reach 90% of steady-state value
    tr = t[tr_idx[0]] - t[0] if tr_idx.size > 0 else np.nan
    if y_ss == 0:
        overshoot = np.nan
    else:
        overshoot = (np.max(y)-y_ss)/y_ss*100
    steady_state_error = np.abs(y_ref - y_ss)
    # ±2% settling
    within = np.abs(y - y_ss) <= 0.02*y_ss
    ts = t[np.where(within)[0][-1]]-t[0] if np.any(within) else np.nan
    return dict(rise_time=tr, overshoot=overshoot,
                settling_time=ts, steady_state_error=steady_state_error)
# ---------- TORQUE RIPPLE ----------
def torque_ripple(torque: np.ndarray) -> float:
    """
    Compute torque ripple as percentage of average torque.
    Parameters
    ----------
    torque : np.ndarray
        Torque signal (Nm)
    """
    Tavg = np.mean(torque)
    return (np.max(torque)-np.min(torque))/(2*Tavg)*100

# ---------- THD ----------
def current_thd(i_signal: np.ndarray, fs: float, plot_fft: bool=False) -> float:
    N=len(i_signal)
    fft_vals=np.fft.fft(i_signal)
    fft_mag=2.0/N*np.abs(fft_vals[:N//2])
    freqs=np.fft.fftfreq(N,1/fs)[:N//2]
    I1=fft_mag[np.argmax(fft_mag)]
    thd=100*np.sqrt(np.sum(fft_mag**2)-I1**2)/I1
    # Plot FFT magnitude in semilog scale if needed
    if plot_fft:
        plt.figure()
        plt.semilogx(freqs, fft_mag)
        title = "Current FFT Magnitude Spectrum - THD: {:.2f}%".format(thd)
        plt.title(title)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (A)")
        plt.grid()
        plt.show()
    return thd

# ---------- TDD ----------
def compute_tdd(i_signal, fs, I_rated, plot_fft=False):
    """
    Compute Total Demand Distortion (TDD) of a current waveform.
    Parameters
    ----------
    i_signal : array_like
        Current samples (A)
    fs : float
        Sampling frequency (Hz)
    I_rated : float
        Rated RMS or peak current used as denominator (A)
    Returns
    -------
    tdd : float
        Total demand distortion in percent of rated current
    freqs, fft_mag : arrays
        Frequency vector and single-sided amplitude spectrum
    """
    N = len(i_signal)
    fft_vals = np.fft.fft(i_signal)
    fft_mag = 2.0 / N * np.abs(fft_vals[:N//2])
    freqs = np.fft.fftfreq(N, 1/fs)[:N//2]

    # Find fundamental (max magnitude excluding DC)
    idx_fund = np.argmax(fft_mag[1:]) + 1
    I1 = fft_mag[idx_fund]
    # RMS of harmonics (excluding DC and fundamental)
    Ih = np.sqrt(np.sum(fft_mag[1:]**2) - I1**2)
    # Total Demand Distortion
    tdd = 100 * Ih / I_rated
    # Plot FFT magnitude in semilog scale if requested
    if plot_fft:
        plt.figure()
        plt.semilogx(freqs, fft_mag)
        title = "Current FFT Magnitude Spectrum - TDD: {:.2f}%".format(tdd)
        plt.title(title)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (A)")
        plt.grid()
        plt.show()
    return tdd

# ---------- EFFICIENCY / PF ----------
def efficiency_powerfactor(vabc: np.ndarray, iabc: np.ndarray) -> tuple:
    """
    Compute efficiency and power factor from three-phase voltages and currents.

    Parameters
    ----------
    vabc : np.ndarray
        Three-phase voltage signals (V), shape [3, N] or [N, 3].
        Must be balanced and sinusoidal for correct results.
    iabc : np.ndarray
        Three-phase current signals (A), shape [3, N] or [N, 3].
        Must be balanced and sinusoidal for correct results.

    Limitations
    -----------
    This function uses the Clarke transform for instantaneous power calculation.
    The Hilbert transform is not used; results may be inaccurate for unbalanced or non-sinusoidal signals.
    """

    v = np.array(vabc)
    i = np.array(iabc)

    # Validate input shapes
    if v.ndim != 2 or i.ndim != 2:
        raise ValueError("Input arrays must be 2D and shaped [3, N] or [N, 3]")
    # Transpose if needed to ensure shape [3, N]
    if v.shape[0] != 3 and v.shape[1] == 3:
        v = v.T
        i = i.T
    elif v.shape[0] != 3 and v.shape[1] != 3:
        raise ValueError("Input arrays must be shaped [3, N] or [N, 3]")

    # Compute instantaneous real power
    p_inst = np.sum(v * i, axis=0)

    # Clarke transform for reactive power calculation
    T = (2/3) * np.array([[1, -0.5, -0.5],
                          [0, np.sqrt(3)/2, -np.sqrt(3)/2]])
    v_alpha = T[0, 0] * v[0] + T[0, 1] * v[1] + T[0, 2] * v[2]
    v_beta  = T[1, 0] * v[0] + T[1, 1] * v[1] + T[1, 2] * v[2]
    i_alpha = T[0, 0] * i[0] + T[0, 1] * i[1] + T[0, 2] * i[2]
    i_beta  = T[1, 0] * i[0] + T[1, 1] * i[1] + T[1, 2] * i[2]

    # Compute instantaneous reactive power (per Clarke transform)
    q_inst = v_alpha * i_beta - v_beta * i_alpha
    P = np.mean(p_inst)
    Q = np.mean(q_inst)
    i_alpha = T[0,0]*i[0] + T[0,1]*i[1] + T[0,2]*i[2]
    i_beta  = T[1,0]*i[0] + T[1,1]*i[1] + T[1,2]*i[2]
    # Instantaneous real and reactive power
    p_inst = v_alpha * i_alpha + v_beta * i_beta
    q_inst = v_beta * i_alpha - v_alpha * i_beta
    P = np.mean(p_inst)
    Q = np.mean(q_inst)
    S = np.sqrt(P**2 + Q**2)
    eff = 100 * P / S if S != 0 else np.nan
    pf = P / S if S != 0 else np.nan
    return eff, pf

# ---------- IEEE 519 HARMONIC ANALYSIS ----------
def harmonic_analysis(i_signal, fs, f_fund, I_rated, isc_il=100):
    """
    Compute harmonic spectrum and check IEEE 519 current limits.
    Parameters
    ----------
    i_signal : array_like
        Current samples (A)
    fs : float
        Sampling frequency (Hz)
    f_fund : float
        Fundamental frequency (Hz)
    I_rated : float
        Rated RMS or peak current (A)
    isc_il : float
        Ratio of short-circuit current to max demand current (ISC/IL),
        determines harmonic limits per IEEE 519.
    Returns
    -------
    harmonics : dict with keys:
        'order', 'amplitude', 'percent_rated', 'limit'
    """

    N = len(i_signal)
    fft_vals = np.fft.fft(i_signal * np.hanning(N))
    fft_mag = 2.0 / N * np.abs(fft_vals[:N // 2])
    freqs = np.fft.fftfreq(N, 1 / fs)[:N // 2]

    # Identify harmonic orders (round frequency / f_fund)
    harmonic_order = np.round(freqs / f_fund).astype(int)
    valid = (harmonic_order >= 1) & (freqs > 0)
    order = harmonic_order[valid]
    amps = fft_mag[valid]
    # Group by harmonic order
    unique_orders = np.unique(order)
    harm_amp = []
    for h in unique_orders:
        mask = order == h
        harm_amp.append(np.sqrt(np.sum(amps[mask] ** 2)))

    harm_amp = np.array(harm_amp)
    percent_rated = 100 * harm_amp / I_rated

    # Fundamental index and harmonics
    h_orders = unique_orders[unique_orders >= 1]
    h_amps = percent_rated[unique_orders >= 1]

    # Compute IEEE 519 limits roughly (per ISC/IL class)
    def ieee_limit(h):
        if isc_il <= 20:
            L = {11: 4.0, 17: 2.0, 23: 1.5, 35: 0.6, 100: 0.3}
        elif isc_il <= 50:
            L = {11: 7.0, 17: 3.5, 23: 2.5, 35: 1.0, 100: 0.5}
        elif isc_il <= 100:
            L = {11: 10.0, 17: 4.5, 23: 4.0, 35: 1.5, 100: 0.7}
        else:
            L = {11: 15.0, 17: 6.0, 23: 5.0, 35: 2.0, 100: 1.0}
        if h < 11:
            return L[11]
        elif h < 17:
            return L[17]
        elif h < 23:
            return L[23]
        elif h < 35:
            return L[35]
        else:
            return L[100]

    limits = np.array([ieee_limit(h) for h in h_orders])

    # Plot
    plt.figure(figsize=(8, 4))
    plt.bar(h_orders, h_amps, width=0.6, label="Measured (%)")
    plt.plot(h_orders, limits, 'r--', label="IEEE 519 Limit (%)")
    plt.xlabel("Harmonic Order h")
    plt.ylabel("Current Magnitude (% of I_rated)")
    plt.title("IEEE 519 Harmonic Current Spectrum")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Return data
    return {
        "order": h_orders,
        "amplitude": harm_amp,
        "percent_rated": h_amps,
        "limit": limits,
    }

# Example usage
if __name__ == "__main__":
    fs = 10000        # sampling frequency (Hz)
    f_fund = 50       # fundamental frequency (Hz)
    t = np.arange(0, 0.2, 1/fs)
    # Simulate current waveform with 5th and 7th harmonics
    i_signal = (10*np.sin(2*np.pi*f_fund*t)
               + 1.5*np.sin(2*np.pi*5*f_fund*t)
               + 1.0*np.sin(2*np.pi*7*f_fund*t))
    I_rated = 10  # A RMS or peak as rated base
    results = harmonic_analysis(i_signal, fs, f_fund, I_rated)

