import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, accuracy_score,
    classification_report, cohen_kappa_score, matthews_corrcoef,
    f1_score
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# -- Ρυθμίσεις -- 

rng         = np.random.default_rng(42)
mod_types   = ['QPSK', '8PSK', '16QAM', '64QAM']
N_bursts    = 500          # bursts ανά διαμόρφωση
BURST_LEN   = 128          # σύμβολα ανά burst
SNR_LEVELS  = [0, 5, 10, 15, 20]   # dB — τρέχει για όλες τις τιμές
h           = np.array([1.0, 0.5, 0.3])


# -- Διαμορφώσεις --

def generate_modulation(mod_type, N):
    if mod_type == 'QPSK':
        bits    = rng.integers(0, 2, 2 * N)
        symbols = (2 * bits[0::2] - 1) + 1j * (2 * bits[1::2] - 1)
        return symbols / np.sqrt(2)
    elif mod_type == '8PSK':
        idx = rng.integers(0, 8, N)
        return np.exp(1j * 2 * np.pi * idx / 8)
    elif mod_type == '16QAM':
        lvl = np.array([-3, -1, 1, 3])
        s   = lvl[rng.integers(0, 4, N)] + 1j * lvl[rng.integers(0, 4, N)]
        return s / np.sqrt(10)
    elif mod_type == '64QAM':
        lvl = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
        s   = lvl[rng.integers(0, 8, N)] + 1j * lvl[rng.integers(0, 8, N)]
        return s / np.sqrt(42)


# 2. -- Θόρυβος AWGN -- 

def add_awgn(signal, snr_db):
    snr_lin      = 10 ** (snr_db / 10)
    signal_power = np.mean(np.abs(signal) ** 2)
    noise_power  = signal_power / snr_lin
    noise = np.sqrt(noise_power / 2) * (
        rng.standard_normal(len(signal)) +
        1j * rng.standard_normal(len(signal))
    )
    return signal + noise


# -- Εξαγωγή χαρακτηριστικών ανά burst --

def extract_features_burst(burst):
    amp       = np.abs(burst)
    phase     = np.angle(burst)
    power     = amp ** 2
    inst_freq = np.concatenate(([0.0], np.diff(np.unwrap(phase))))
    real      = burst.real
    imag      = burst.imag

    def stats4(x):
        return [np.mean(x), np.var(x), skew(x), kurtosis(x)]

    return np.array(
        stats4(amp) + stats4(power) + stats4(inst_freq) +
        stats4(real) + stats4(imag) +
        [np.mean(amp**3), np.mean(amp**4), np.mean(amp**6), np.mean(amp**8),
         np.max(power) / (np.mean(power) + 1e-12),
         np.var(real) / (np.var(imag) + 1e-12)],
        dtype=np.float32
    )


# -- Dataset builder --
def build_dataset(snr_db):
    X, y = [], []
    for mod in mod_types:
        for _ in range(N_bursts):
            symbols = generate_modulation(mod, BURST_LEN)
            rx      = convolve(symbols, h, mode='same')
            rx      = add_awgn(rx, snr_db)
            X.append(extract_features_burst(rx))
            y.append(mod)
    return np.array(X), np.array(y)


# -- Ορισμός ταξινομητών (δημιουργία fresh για κάθε SNR) --

def make_models():
    knn = Pipeline([('sc', StandardScaler()),
                    ('clf', KNeighborsClassifier(n_neighbors=7))])
    svm = Pipeline([('sc', StandardScaler()),
                    ('clf', SVC(kernel='rbf', C=10, gamma='scale',
                                probability=True, random_state=42))])
    rf  = Pipeline([('sc', StandardScaler()),
                    ('clf', RandomForestClassifier(n_estimators=300,
                                                   random_state=42, n_jobs=-1))])
    voting = VotingClassifier(
        estimators=[('knn', knn), ('svm', svm), ('rf', rf)],
        voting='soft'
    )
    return {"KNN": knn, "SVM": svm, "RF": rf, "Voting": voting}


# -- Κύριος βρόχος: πλήρες evaluation ανά SNR level --

# -- Αποθήκευση accuracy για κάθε μοντέλο × SNR --
all_results = {snr: {} for snr in SNR_LEVELS}

for snr in SNR_LEVELS:
    print(f"\n{'='*65}")
    print(f"  SNR = {snr} dB")
    print(f"{'='*65}")

    X, y = build_dataset(snr)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    models = make_models()

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc   = accuracy_score(y_test, y_pred)
        f1m   = f1_score(y_test, y_pred, average='macro')
        kappa = cohen_kappa_score(y_test, y_pred)
        mcc   = matthews_corrcoef(y_test, y_pred)

        all_results[snr][name] = dict(accuracy=acc, f1_macro=f1m,
                                      cohen_kappa=kappa, mcc=mcc,
                                      y_pred=y_pred)

        print(f"  [{name:6s}]  Acc: {acc:.2%}  F1: {f1m:.4f}  "
              f"κ: {kappa:.4f}  MCC: {mcc:.4f}")

    # --- Confusion matrices για αυτό το SNR (2×2 grid) ---
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(f'Confusion Matrices — SNR = {snr} dB', fontsize=14)
    for ax, name in zip(axes.flat, models.keys()):
        yp  = all_results[snr][name]['y_pred']
        acc = all_results[snr][name]['accuracy']
        cm  = confusion_matrix(y_test, yp, labels=mod_types)
        ConfusionMatrixDisplay(cm, display_labels=mod_types).plot(
            cmap='Blues', ax=ax, colorbar=False
        )
        ax.set_title(f'{name} — {acc:.2%}')
    plt.tight_layout()
    plt.savefig(f'cm_SNR{snr}dB.png', dpi=150)
    plt.show()


# -- Accuracy vs SNR — όλοι οι ταξινομητές --

model_names = list(make_models().keys())
markers     = ['o', 's', '^', 'D']
colors      = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
fig.suptitle('Μετρικές ταξινόμησης vs SNR', fontsize=14)
metric_keys    = ['accuracy', 'f1_macro', 'cohen_kappa', 'mcc']
metric_labels  = ['Accuracy', 'F1-score (macro)', "Cohen's Kappa (κ)", 'MCC']

for ax, mkey, mlabel in zip(axes.flat, metric_keys, metric_labels):
    for name, mk, col in zip(model_names, markers, colors):
        vals = [all_results[snr][name][mkey] for snr in SNR_LEVELS]
        ax.plot(SNR_LEVELS, vals, marker=mk, color=col, label=name)
    ax.set_title(mlabel)
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel(mlabel)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('metrics_vs_snr.png', dpi=150)
plt.show()


# -- Πινακας αποτελεσματων --

print(f"\n{'='*65}")
print("  ACCURACY ανά SNR και Ταξινομητή")
print(f"{'='*65}")
header = f"{'SNR (dB)':<10}" + "".join(f"{n:>10}" for n in model_names)
print(header)
print('-' * len(header))
for snr in SNR_LEVELS:
    row = f"{snr:<10}"
    for name in model_names:
        row += f"{all_results[snr][name]['accuracy']:>10.2%}"
    print(row)