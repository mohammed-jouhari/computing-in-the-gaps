"""Physical-layer models used by the seabed-to-orbit simulator.

Acoustic links follow the frequency-dependent path-loss model of Stojanovic
(spreading factor k = 1.5 and Thorp absorption) and the empirical ambient-noise
model (turbulence, shipping, wind, thermal) used in the same work. The mean
in-band SNR selects one of four modem modes; a MAC efficiency factor accounts
for handshakes and guard times that cover the long propagation delay.
"""
from __future__ import annotations

import numpy as np

SOUND_SPEED = 1500.0  # m/s


def thorp_db_per_km(f_khz):
    """Thorp absorption coefficient in dB/km, f in kHz."""
    f2 = np.asarray(f_khz, dtype=float) ** 2
    return 0.11 * f2 / (1 + f2) + 44 * f2 / (4100 + f2) + 2.75e-4 * f2 + 0.003


def noise_psd_db(f_khz, shipping=0.5, wind_mps=5.0):
    """Ambient noise PSD in dB re 1 uPa^2/Hz (turbulence, shipping, wind, thermal)."""
    f = np.asarray(f_khz, dtype=float)
    nt = 17 - 30 * np.log10(f)
    ns = 40 + 20 * (shipping - 0.5) + 26 * np.log10(f) - 60 * np.log10(f + 0.03)
    nw = 50 + 7.5 * np.sqrt(wind_mps) + 20 * np.log10(f) - 40 * np.log10(f + 0.4)
    nth = -15 + 20 * np.log10(f)
    return 10 * np.log10(10 ** (nt / 10) + 10 ** (ns / 10) + 10 ** (nw / 10) + 10 ** (nth / 10))


def path_loss_db(dist_km, f_khz, k=1.5):
    """10 log10 A(l, f) with l in km (reference distance 1 m)."""
    return k * 10 * np.log10(dist_km * 1000.0) + dist_km * thorp_db_per_km(f_khz)


def mean_inband_snr_db(dist_km, band_khz=(18.0, 34.0), source_level_db=175.0,
                       shipping=0.5, wind_mps=5.0):
    """Average SNR over the modem band for a flat transmit spectrum."""
    f = np.linspace(band_khz[0], band_khz[1], 200)
    psd_tx = source_level_db - 10 * np.log10((band_khz[1] - band_khz[0]) * 1e3)
    snr = psd_tx - path_loss_db(dist_km, f) - noise_psd_db(f, shipping, wind_mps)
    return float(10 * np.log10(np.mean(10 ** (snr / 10))))


# Modem modes: (nominal bit rate in b/s, required mean SNR in dB). The thresholds
# include an implementation margin for multipath and Doppler spreading.
MODEM_MODES = ((9600.0, 25.0), (4800.0, 20.0), (2400.0, 15.0), (1200.0, 10.0))


def acoustic_goodput_bps(dist_km, mac_efficiency=0.7, **kw):
    """Goodput of the adaptive acoustic modem at a given range (0 if no mode works)."""
    snr = mean_inband_snr_db(dist_km, **kw)
    for rate, thr in MODEM_MODES:
        if snr >= thr:
            return rate * mac_efficiency
    return 0.0


def acoustic_delay_s(dist_km):
    return dist_km * 1000.0 / SOUND_SPEED


if __name__ == "__main__":
    for d in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
        print(f"{d:4.1f} km  SNR {mean_inband_snr_db(d):5.1f} dB  goodput {acoustic_goodput_bps(d)/1e3:4.2f} kb/s")
