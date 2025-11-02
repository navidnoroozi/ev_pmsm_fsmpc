# === performance_ieee519.py ===
import numpy as np
import matplotlib.pyplot as plt

# ---------- IEEE 519 limit helper ----------
def _ieee519_individual_limit(h, isc_il):
    """
    Piecewise limits per IEEE-519 tables (current distortion) as % of I_rated.
    This simplified mapping follows typical buckets by ISC/IL.
    """
    # Buckets (choose the one that matches your ISC/IL)
    if isc_il <= 20:
        L = {11: 4.0, 17: 2.0, 23: 1.5, 35: 0.6, 100: 0.3}
    elif isc_il <= 50:
        L = {11: 7.0, 17: 3.5, 23: 2.5, 35: 1.0, 100: 0.5}
    elif isc_il <= 100:
        L = {11: 10.0, 17: 4.5, 23: 4.0, 35: 1.5, 100: 0.7}
    elif isc_il <= 1000:
        L = {11: 12.0, 17: 5.5, 23: 5.0, 35: 2.0, 100: 1.0}
    else:
        # very strong grid
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

# ---------- core spectrum & harmonics ----------
def _harmonics_percent_of_rated(i_signal, fs, f_fund, I_rated):
    """
    Returns (orders, amps_abs, amps_pct), where amps are single-sided
    harmonic magnitudes aggregated by integer order, in A and % of I_rated.
    """
    i = np.asarray(i_signal).flatten()
    N = len(i)
    if N < 8:
        raise ValueError("Signal too short for FFT")

    # Window to reduce leakage
    w = np.hanning(N)
    xw = i * w
    # Single-sided spectrum
    fft_vals = np.fft.fft(xw)
    mag = (2.0 / np.sum(w)) * np.abs(fft_vals[:N // 2])  # amplitude correction with window sum
    freqs = np.fft.fftfreq(N, d=1.0 / fs)[:N // 2]

    # Map bins to nearest harmonic order
    orders_all = np.round(freqs / f_fund).astype(int)
    valid = (freqs > 0) & (orders_all >= 1)
    orders = orders_all[valid]
    mags = mag[valid]

    # Group energy per integer harmonic order
    uniq = np.unique(orders)
    amps = []
    for h in uniq:
        mask = orders == h
        # root-sum-square across bins of same order (guarding small leakage)
        amps.append(np.sqrt(np.sum(mags[mask] ** 2)))
    amps = np.array(amps)  # linear magnitude in A (approx)
    amps_pct = 100.0 * amps / float(I_rated)

    return uniq, amps, amps_pct

def _thd_percent_from_orders(orders, amps):
    """THD in % using fundamental magnitude as base."""
    if len(orders) == 0:
        return np.nan
    # fundamental assumed to be order == 1
    try:
        I1 = float(amps[orders == 1])
    except Exception:
        return np.nan
    if I1 <= 1e-12:
        return np.nan
    Ih = np.sqrt(np.sum(amps[orders >= 2] ** 2))
    return 100.0 * Ih / I1

def _tdd_percent_from_orders(amps, I_rated):
    """TDD in % using I_rated as base."""
    Ih = np.sqrt(np.sum(amps[1:] ** 2)) if len(amps) > 1 else 0.0
    return 100.0 * Ih / float(I_rated)

# ---------- three-phase analysis ----------
def analyze_ieee519_3ph(i_a, i_b, i_c, fs, f_fund, I_rated, isc_il=100,
                        max_harm=50, do_plot=True, title="IEEE 519 Harmonic Analysis (3-Phase)"):
    """
    Three-phase harmonic analysis with IEEE-519 limits.

    Parameters
    ----------
    i_a, i_b, i_c : arrays
        Phase currents (A)
    fs : float
        Sampling frequency (Hz)
    f_fund : float
        Fundamental frequency (Hz)
    I_rated : float
        Rated RMS (preferred) or rated amplitude per phase (A)
    isc_il : float
        Short-circuit ratio at PCC
    max_harm : int
        Highest harmonic order to report/plot
    do_plot : bool
        If True, makes a dashboard bar plot
    title : str
        Plot title

    Returns
    -------
    results : dict with per-phase and aggregate metrics
    """
    # Per phase spectra
    ord_a, ampsA, pctA = _harmonics_percent_of_rated(i_a, fs, f_fund, I_rated)
    ord_b, ampsB, pctB = _harmonics_percent_of_rated(i_b, fs, f_fund, I_rated)
    ord_c, ampsC, pctC = _harmonics_percent_of_rated(i_c, fs, f_fund, I_rated)

    # Make common order grid up to max_harm
    h = np.arange(1, max_harm + 1, dtype=int)

    def regrid(ord_src, vec_src):
        out = np.zeros_like(h, dtype=float)
        for k, hk in enumerate(h):
            if hk in ord_src:
                out[k] = float(vec_src[ord_src == hk])
        return out

    ampsA_h = regrid(ord_a, ampsA); pctA_h = regrid(ord_a, pctA)
    ampsB_h = regrid(ord_b, ampsB); pctB_h = regrid(ord_b, pctB)
    ampsC_h = regrid(ord_c, ampsC); pctC_h = regrid(ord_c, pctC)

    # THD (per-phase)
    thd_A = _thd_percent_from_orders(ord_a, ampsA)
    thd_B = _thd_percent_from_orders(ord_b, ampsB)
    thd_C = _thd_percent_from_orders(ord_c, ampsC)

    # TDD (per-phase) using I_rated base
    tdd_A = _tdd_percent_from_orders(ampsA, I_rated)
    tdd_B = _tdd_percent_from_orders(ampsB, I_rated)
    tdd_C = _tdd_percent_from_orders(ampsC, I_rated)

    # IEEE-519 individual limits curve for selected ISC/IL
    limits = np.array([_ieee519_individual_limit(hh, isc_il) for hh in h], dtype=float)

    # Aggregate (average) metrics
    thd_avg = np.nanmean([thd_A, thd_B, thd_C])
    tdd_avg = np.nanmean([tdd_A, tdd_B, tdd_C])

    results = {
        "orders": h,
        "per_phase": {
            "A": {"amps_A": ampsA_h, "pct_A": pctA_h, "THD%": thd_A, "TDD%": tdd_A},
            "B": {"amps_B": ampsB_h, "pct_B": pctB_h, "THD%": thd_B, "TDD%": tdd_B},
            "C": {"amps_C": ampsC_h, "pct_C": pctC_h, "THD%": thd_C, "TDD%": tdd_C},
        },
        "limits_percent_of_rated": limits,
        "THD_avg%": thd_avg,
        "TDD_avg%": tdd_avg,
        "isc_il": isc_il,
        "I_rated": I_rated,
        "f_fund": f_fund,
    }

    if do_plot:
        fig, axs = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
        for ax, pct, ph in zip(
            axs,
            [pctA_h, pctB_h, pctC_h],
            ["Phase A", "Phase B", "Phase C"]
        ):
            ax.bar(h, pct, width=0.7, label=f"{ph} harmonics (% of I_rated)")
            ax.plot(h, limits, 'r--', linewidth=2, label="IEEE 519 limit")
            ax.set_ylabel("% of I_rated")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(loc="upper right")

        axs[-1].set_xlabel("Harmonic order h")
        fig.suptitle(f"{title}\nISC/IL={isc_il},  THD_avg={thd_avg:.2f}%,  TDD_avg={tdd_avg:.2f}%")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Optional: add a compact table of THD/TDD per phase
        text = (f"A: THD={thd_A:.2f}%, TDD={tdd_A:.2f}%   |   "
                f"B: THD={thd_B:.2f}%, TDD={tdd_B:.2f}%   |   "
                f"C: THD={thd_C:.2f}%, TDD={tdd_C:.2f}%")
        fig.text(0.5, 0.005, text, ha="center", va="bottom")

        plt.show()

    return results

# ---------- demo ----------
if __name__ == "__main__":
    fs = 10000.0
    f1 = 50.0
    t = np.arange(0, 0.25, 1/fs)

    # Synthetic 3-phase with 5th and 7th harmonics
    I1 = 10.0
    iA = I1*np.sqrt(2)*np.sin(2*np.pi*f1*t) \
       + 0.8*np.sqrt(2)*np.sin(2*np.pi*5*f1*t) \
       + 0.5*np.sqrt(2)*np.sin(2*np.pi*7*f1*t)
    iB = I1*np.sqrt(2)*np.sin(2*np.pi*f1*t - 2*np.pi/3) \
       + 0.6*np.sqrt(2)*np.sin(2*np.pi*5*f1*t - 2*np.pi/3) \
       + 0.4*np.sqrt(2)*np.sin(2*np.pi*7*f1*t - 2*np.pi/3)
    iC = I1*np.sqrt(2)*np.sin(2*np.pi*f1*t + 2*np.pi/3) \
       + 0.7*np.sqrt(2)*np.sin(2*np.pi*5*f1*t + 2*np.pi/3) \
       + 0.3*np.sqrt(2)*np.sin(2*np.pi*7*f1*t + 2*np.pi/3)

    I_rated = I1  # choose your rated RMS per phase here
    analyze_ieee519_3ph(iA, iB, iC, fs=fs, f_fund=f1, I_rated=I_rated, isc_il=100)