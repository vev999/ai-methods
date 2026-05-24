import numpy as np

with open('Projekt/winequality-red.csv') as f:
    header = f.readline().strip().split(',')

data = np.loadtxt('Projekt/winequality-red.csv', delimiter=',', skiprows=1)
feature_names = header[:-1]
X = data[:, :-1]
y = data[:, -1].astype(int)

print("=== ANALIZA ZBIORU DANYCH ===")
print(f"Rozmiar: {data.shape[0]} probek, {len(feature_names)} cech")
print(f"Cechy: {feature_names}")

print(f"\nRozklad klasy 'quality':")
for val in sorted(np.unique(y)):
    cnt = np.sum(y == val)
    print(f"  jakosc {val}: {cnt:4d} probek ({cnt/len(y)*100:.1f}%)")

print(f"\nPodstawowe statystyki:")
print(f"{'Cecha':<25} {'mean':>8} {'std':>8} {'min':>8} {'25%':>8} {'50%':>8} {'75%':>8} {'max':>8}")
print("-" * 85)
for i, name in enumerate(feature_names):
    col = X[:, i]
    print(f"{name:<25} {np.mean(col):>8.3f} {np.std(col):>8.3f} {np.min(col):>8.3f} "
          f"{np.percentile(col, 25):>8.3f} {np.percentile(col, 50):>8.3f} "
          f"{np.percentile(col, 75):>8.3f} {np.max(col):>8.3f}")
