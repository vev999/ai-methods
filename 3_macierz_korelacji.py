import numpy as np
import matplotlib.pyplot as plt

with open('Projekt/winequality-red.csv') as f:
    header = f.readline().strip().split(',')

data = np.loadtxt('Projekt/winequality-red.csv', delimiter=',', skiprows=1)
col_names = header

corr = np.corrcoef(data.T)

fig, ax = plt.subplots(figsize=(13, 11))
im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(im, ax=ax)

ax.set_xticks(range(len(col_names)))
ax.set_yticks(range(len(col_names)))
ax.set_xticklabels(col_names, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(col_names, fontsize=9)

for i in range(len(col_names)):
    for j in range(len(col_names)):
        ax.text(j, i, f'{corr[i, j]:.2f}',
                ha='center', va='center', fontsize=7,
                color='white' if abs(corr[i, j]) > 0.6 else 'black')

ax.set_title('Macierz korelacji Pearsona', fontsize=13)
plt.tight_layout()
plt.savefig('Projekt/3_macierz_korelacji.png', dpi=100)
print("[OK] Saved: 3_macierz_korelacji.png")
