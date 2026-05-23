import  numpy                       as np
from    scipy.stats                 import wilcoxon
from    tabulate                    import tabulate
from    sklearn.naive_bayes         import GaussianNB
from    sklearn.preprocessing       import StandardScaler
from    sklearn.neighbors           import KNeighborsClassifier
from    sklearn.tree                import DecisionTreeClassifier
from    sklearn.model_selection     import RepeatedStratifiedKFold
from    sklearn.metrics             import confusion_matrix, balanced_accuracy_score, accuracy_score, f1_score

np.set_printoptions(precision=3, suppress=True)

data = np.genfromtxt('winequality-red.csv', delimiter=',', dtype=float, skip_header=1)
header = np.loadtxt('winequality-red.csv', delimiter=',', dtype=str, max_rows=1)

X = data[:,:-1]
y = data[:,-1]

clfs = [GaussianNB(), KNeighborsClassifier(), DecisionTreeClassifier()]
clfs_names = ["GNB", "KNN", "DT"]
metrics = [accuracy_score, balanced_accuracy_score, f1_score]
metrics_names = ["AC", "BAC", "F1-SCORE"]
n_repeats = 10
n_splits = 5
n_folds = n_repeats * n_splits
rsfk = RepeatedStratifiedKFold(n_repeats=n_repeats, n_splits=n_splits)
y_preds = [[], [], []]
y_tests = [[], [], []]
alpha = 0.05
significant = False
# Score[classificator, metric, fold]
score = np.zeros(shape=(len(clfs), len(metrics), n_folds)) 

# Test with data normalization - utilizing StandardScaler
scaler = StandardScaler()

for fold, (train_index, test_index) in enumerate(rsfk.split(X=X, y=y)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    scaler.fit(X=X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for clf_index, clf in enumerate(clfs):
        clf.fit(X=X_train_scaled, y=y_train)
        y_pred = clf.predict(X_test_scaled)

        for metric_index, metric in enumerate(metrics): 
            if metric == f1_score:
                metric_score = metric(y_true=y_test, y_pred=y_pred, average='macro')
            else:
                metric_score = metric(y_true=y_test, y_pred=y_pred)
            score[clf_index][metric_index][fold] = metric_score
    
        y_preds[clf_index].extend(y_pred)
        y_tests[clf_index].extend(y_test)

headers = ["", "GNB", "KNN", "DT"]
rows = [
    ["AC", f"{score[0][0][:].mean():.3} +- {score[0][0][:].std():.3}",  f"{score[1][0][:].mean():.3} +- {score[1][0][:].std():.3}",  f"{score[2][0][:].mean():.3} +- {score[2][0][:].std():.3}"],
    ["BAC", f"{score[0][1][:].mean():.3} +- {score[0][1][:].std():.3}",  f"{score[1][1][:].mean():.3} +- {score[1][1][:].std():.3}",  f"{score[2][1][:].mean():.3} +- {score[2][1][:].std():.3}"],
    ["F1-SCORE", f"{score[0][2][:].mean():.3} +- {score[0][2][:].std():.3}",  f"{score[1][2][:].mean():.3} +- {score[1][2][:].std():.3}",  f"{score[2][2][:].mean():.3} +- {score[2][2][:].std():.3}"],
]

for clf_index, clf in enumerate(clfs):
    print(f"Confusion matrix {clfs_names[clf_index]}:")
    print(f"{confusion_matrix(y_true=y_tests[clf_index], y_pred=y_preds[clf_index], normalize='true')}")

print("-----------------------------------------------------------")

print(tabulate(rows, headers=headers, floatfmt=".3f"))

print("-----------------------------------------------------------")

for clf_index_1, clf_1 in enumerate(clfs):
    for clf_index_2, clf_2 in enumerate(clfs):
        if clf_index_1 < clf_index_2:
            # Metric used in wilcoxon test is F1-score
            wilcoxon_test = wilcoxon(score[clf_index_1][2][:], score[clf_index_2][2][:])
            if wilcoxon_test.pvalue < alpha:
                significant = ''
            else:
                significant = 'not '
            print(f"Difference between {clfs_names[clf_index_1]} and {clfs_names[clf_index_2]} is {significant}statistically significant.")