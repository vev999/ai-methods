import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold

with open('Projekt/winequality-red.csv') as f:
    header = f.readline().strip().split(',')

data = np.loadtxt('Projekt/winequality-red.csv', delimiter=',', skiprows=1)
feature_names = header[:-1]
X = data[:, :-1]
y = data[:, -1].astype(int)

vt = VarianceThreshold()
X_vt = vt.fit_transform(X)

skb = SelectKBest(f_classif, k='all')
skb.fit(X_vt, y)
f_scores = skb.scores_
p_values = skb.pvalues_

sorted_idx = np.argsort(f_scores)[::-1]

print("=== ANALIZA ISTOTNOSCI CECH (SelectKBest / f_classif) ===")
print(f"\n{'Rank':<5} {'Cecha':<30} {'F-score':>10} {'p-value':>12}")
print("-" * 60)
for rank, i in enumerate(sorted_idx):
    sig = "*" if p_values[i] < 0.05 else ""
    print(f"{rank+1:<5} {feature_names[i]:<30} {f_scores[i]:>10.2f} {p_values[i]:>12.6f} {sig}")

fig, ax = plt.subplots(figsize=(12, 6))
sorted_features = [feature_names[i] for i in sorted_idx]
sorted_scores = f_scores[sorted_idx]
sorted_pvals = p_values[sorted_idx]

bar_colors = ['steelblue' if p < 0.05 else 'lightgray' for p in sorted_pvals]
ax.bar(range(len(sorted_features)), sorted_scores, color=bar_colors, edgecolor='k')
ax.set_xticks(range(len(sorted_features)))
ax.set_xticklabels(sorted_features, rotation=45, ha='right')
ax.set_ylabel('F-score')
ax.set_title('Istotnosc cech (f_classif) – niebieski = istotna statystycznie (p<0.05)')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('Projekt/4_istotnosc_fclassif.png', dpi=100)
print("[OK] Saved: 4_istotnosc_fclassif.png")
