import numpy as np
import matplotlib.pyplot as plt

with open('Projekt/winequality-red.csv') as f:
    header = f.readline().strip().split(',')

data = np.loadtxt('Projekt/winequality-red.csv', delimiter=',', skiprows=1)
feature_names = header[:-1]
X = data[:, :-1]
y = data[:, -1].astype(int)

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

for i, name in enumerate(feature_names):
    axes[i].hist(X[:, i], bins=30, color='steelblue', edgecolor='k', alpha=0.7)
    axes[i].set_title(name)
    axes[i].set_xlabel('Wartosc')
    axes[i].set_ylabel('Licznosc')
    axes[i].grid(True, alpha=0.3)

axes[-1].hist(y, bins=range(3, 10), color='darkorange', edgecolor='k', alpha=0.7, align='left')
axes[-1].set_title('quality (zmienna docelowa)')
axes[-1].set_xlabel('Ocena')
axes[-1].set_ylabel('Licznosc')
axes[-1].grid(True, alpha=0.3)

plt.suptitle('Rozklady cech – Red Wine Quality', fontsize=14)
plt.tight_layout()
plt.savefig('Projekt/2_histogramy.png', dpi=100)
print("[OK] Saved: 2_histogramy.png")
