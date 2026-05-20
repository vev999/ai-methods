import numpy as np
import matplotlib.pyplot as plt

with open('Projekt/winequality-red.csv') as f:
    header = f.readline().strip().split(',')

data = np.loadtxt('Projekt/winequality-red.csv', delimiter=',', skiprows=1)
feature_names = header[:-1]

corr_full = np.corrcoef(data.T)
corr_with_quality = corr_full[:-1, -1]

sorted_idx = np.argsort(np.abs(corr_with_quality))[::-1]
sorted_features = [feature_names[i] for i in sorted_idx]
sorted_corrs = corr_with_quality[sorted_idx]

print("=== KORELACJA PEARSONA CECH Z 'quality' ===")
print(f"\n{'Cecha':<30} {'r Pearsona':>12}")
print("-" * 45)
for feat, r in zip(sorted_features, sorted_corrs):
    print(f"{feat:<30} {r:>12.4f}")

fig, ax = plt.subplots(figsize=(12, 6))
colors = ['green' if v > 0 else 'tomato' for v in sorted_corrs]
ax.bar(range(len(sorted_features)), sorted_corrs, color=colors, edgecolor='k')
ax.set_xticks(range(len(sorted_features)))
ax.set_xticklabels(sorted_features, rotation=45, ha='right')
ax.set_ylabel('Wspolczynnik korelacji Pearsona')
ax.set_title("Korelacja cech z 'quality' (zielony = pozytywna, czerwony = negatywna)")
ax.axhline(0, color='black', linewidth=0.8)
ax.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(sorted_corrs):
    ax.text(i, v + (0.005 if v >= 0 else -0.025), f'{v:.3f}', ha='center', fontsize=8)
plt.tight_layout()
plt.savefig('Projekt/5_pearson.png', dpi=100)
print("[OK] Saved: 5_pearson.png")
