import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from numpy.random import default_rng


# Ρυθμίσεις
N       = 1000          # αριθμός συμβόλων
rng     = default_rng(42)
snr_db_list = [20, 15, 10, 0, -10]

# Κρουστική απόκριση καναλιού (Multipath)
h = np.array([1.0, 0.5, 0.3])


# ---- Γεννήτριες Συμβόλων ----------------


def generate_bpsk(N, rng):
    """1 bit/σύμβολο → ±1"""
    bits = rng.integers(0, 2, size=N)
    return (2 * bits - 1).astype(complex)          # {-1, +1}

def generate_qpsk(N, rng):
    """2 bits/σύμβολο → Gray QPSK, ±1 ±j"""
    bits = rng.integers(0, 2, size=2 * N)
    return (2 * bits[0::2] - 1) + 1j * (2 * bits[1::2] - 1)

def generate_16qam(N, rng):
    """4 bits/σύμβολο → 16-QAM, αστερισμός ±1,±3"""
    bits = rng.integers(0, 2, size=4 * N)
    # Gray-coded αντιστοίχηση 2-bit → {-3,-1,+1,+3}
    gray2val = {(0,0): -3, (0,1): -1, (1,1): 1, (1,0): 3}
    I = np.array([gray2val[(bits[4*i],   bits[4*i+1])] for i in range(N)], dtype=float)
    Q = np.array([gray2val[(bits[4*i+2], bits[4*i+3])] for i in range(N)], dtype=float)
    symbols = (I + 1j * Q) / np.sqrt(10)           # κανονικοποίηση: E[|s|²]=1
    return symbols

def generate_64qam(N, rng):
    """6 bits/σύμβολο → 64-QAM, αστερισμός ±1,±3,±5,±7"""
    bits = rng.integers(0, 2, size=6 * N)
    # Gray-coded αντιστοίχηση 3-bit → {-7,-5,-3,-1,+1,+3,+5,+7}
    gray3val = {
        (0,0,0): -7, (0,0,1): -5, (0,1,1): -3, (0,1,0): -1,
        (1,1,0):  1, (1,1,1):  3, (1,0,1):  5, (1,0,0):  7
    }
    I = np.array([gray3val[(bits[6*i],   bits[6*i+1], bits[6*i+2])] for i in range(N)], dtype=float)
    Q = np.array([gray3val[(bits[6*i+3], bits[6*i+4], bits[6*i+5])] for i in range(N)], dtype=float)
    symbols = (I + 1j * Q) / np.sqrt(42)           # κανονικοποίηση: E[|s|²]=1
    return symbols


# ---- Κανάλι & Θόρυβος -------------------

def apply_multipath(symbols, h):
    """Συνέλιξη με κρουστική απόκριση καναλιού."""
    y = convolve(symbols, h, mode='full')
    return y[:len(symbols)]                         # ίδιο μέγεθος με είσοδο

def add_awgn(signal, snr_db, rng):
    """Προσθέτει μιγαδικό AWGN με βάση το επιθυμητό SNR (dB)."""
    snr_lin     = 10 ** (snr_db / 10)
    sig_power   = np.mean(np.abs(signal) ** 2)
    noise_power = sig_power / snr_lin
    noise = np.sqrt(noise_power / 2) * (
        rng.standard_normal(len(signal)) +
        1j * rng.standard_normal(len(signal))
    )
    return signal + noise


# ---- Ορισμός Διαμορφώσεων ---------------
modulations = [
    {"name": "BPSK",   "generator": generate_bpsk,   "color": "royalblue",   "marker": "o"},
    {"name": "QPSK",   "generator": generate_qpsk,   "color": "tomato",      "marker": "o"},
    {"name": "16-QAM", "generator": generate_16qam,  "color": "mediumseagreen","marker":"o"},
    {"name": "64-QAM", "generator": generate_64qam,  "color": "darkorchid",  "marker": "o"},
]


# ---- Σχεδίαση: ένα figure ανά διαμόρφωση
for mod in modulations:
    name      = mod["name"]
    gen       = mod["generator"]
    color     = mod["color"]

    symbols = gen(N, rng)

    # Αριθμός subplots: 1 αρχικός + 5 SNR = 6  (2×3 grid)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        f"Διαγράμματα Αστερισμού — {name}  |  Multipath h=[1.0, 0.5, 0.3] + AWGN",
        fontsize=14, fontweight='bold'
    )

    # --- 1ο subplot: αρχικά σύμβολα ---
    ax0 = axes[0, 0]
    ax0.scatter(symbols.real, symbols.imag, s=18, alpha=0.55, color=color, edgecolors='none')
    ax0.set_title(f"{name} — Αρχικά Σύμβολα\n(χωρίς κανάλι)", fontsize=10)
    ax0.set_xlabel("In-Phase (I)")
    ax0.set_ylabel("Quadrature (Q)")
    ax0.grid(True, linestyle='--', alpha=0.55)
    ax0.axhline(0, color='k', lw=0.7)
    ax0.axvline(0, color='k', lw=0.7)

    # --- Υπόλοιπα 5 subplots: Multipath + AWGN ---
    snr_colors = ['#e63946','#f4a261','#2a9d8f','#457b9d','#6d6875']
    flat_axes  = [axes[0,1], axes[0,2], axes[1,0], axes[1,1], axes[1,2]]

    for ax, snr_db, sc in zip(flat_axes, snr_db_list, snr_colors):
        y_ch  = apply_multipath(symbols, h)
        y_rx  = add_awgn(y_ch, snr_db, rng)

        ax.scatter(y_rx.real, y_rx.imag, s=8, alpha=0.40, color=sc, edgecolors='none')
        ax.set_title(f"SNR = {snr_db:+} dB  (Multipath + AWGN)", fontsize=10)
        ax.set_xlabel("In-Phase (I)")
        ax.set_ylabel("Quadrature (Q)")
        ax.grid(True, linestyle='--', alpha=0.55)
        ax.axhline(0, color='k', lw=0.7)
        ax.axvline(0, color='k', lw=0.7)

    plt.tight_layout()
    plt.show()


# ---- Συνοπτικό Figure (4×5 grid) --------

fig2, axes2 = plt.subplots(4, 6, figsize=(28, 18))
fig2.suptitle(
    "Συνοπτικά Διαγράμματα Αστερισμού — BPSK / QPSK / 16-QAM / 64-QAM\n"
    "Multipath h=[1.0, 0.5, 0.3] + AWGN",
    fontsize=15, fontweight='bold'
)

snr_colors_sum = ['#1d3557','#e63946','#f4a261','#2a9d8f','#6d6875']

for row_idx, mod in enumerate(modulations):
    name  = mod["name"]
    gen   = mod["generator"]
    color = mod["color"]
    symbols = gen(N, rng)

    # Col 0: αρχικά
    ax = axes2[row_idx, 0]
    ax.scatter(symbols.real, symbols.imag, s=12, alpha=0.6, color=color, edgecolors='none')
    ax.set_title(f"{name}\nΑρχικά", fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.axhline(0, color='k', lw=0.6); ax.axvline(0, color='k', lw=0.6)
    ax.set_xlabel("I"); ax.set_ylabel("Q")

    # Cols 1-5: SNR τιμές
    for col_idx, (snr_db, sc) in enumerate(zip(snr_db_list, snr_colors_sum), start=1):
        ax = axes2[row_idx, col_idx]
        y_ch = apply_multipath(symbols, h)
        y_rx = add_awgn(y_ch, snr_db, rng)
        ax.scatter(y_rx.real, y_rx.imag, s=6, alpha=0.35, color=sc, edgecolors='none')
        ax.set_title(f"{name}\nSNR={snr_db:+}dB", fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.axhline(0, color='k', lw=0.6); ax.axvline(0, color='k', lw=0.6)
        ax.set_xlabel("I"); ax.set_ylabel("Q")

plt.tight_layout()
plt.show()