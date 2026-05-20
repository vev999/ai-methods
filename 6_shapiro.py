import numpy as np
from scipy.stats import shapiro

with open('Projekt/winequality-red.csv') as f:
    header = f.readline().strip().split(',')

data = np.loadtxt('Projekt/winequality-red.csv', delimiter=',', skiprows=1)
feature_names = header[:-1]
X = data[:, :-1]

alpha = 0.05

print("=== TESTY NORMALNOSCI SHAPIRO-WILK (alpha=0.05) ===")
print(f"\n{'Cecha':<30} {'statistic':>10} {'p-value':>12} {'Normalny?':>10}")
print("-" * 65)
for i, name in enumerate(feature_names):
    cecha = np.copy(X[:, i])
    cecha -= np.mean(cecha)
    cecha /= np.std(cecha)
    stat, p = shapiro(X[:, i])
    normal = p > alpha
    print(f"{name:<30} {stat:>10.4f} {p:>12.6f} {'TAK' if normal else 'NIE':>10}")
