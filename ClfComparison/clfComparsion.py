import  numpy                        as np
from    tabulate                     import tabulate
from    sklearn.naive_bayes          import GaussianNB
from    sklearn.neighbors            import KNeighborsClassifier
from    sklearn.tree                 import DecisionTreeClassifier
from    sklearn.model_selection      import RepeatedStratifiedKFold
from    sklearn.metrics              import confusion_matrix, balanced_accuracy_score, accuracy_score, f1_score

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
# Score[classificator, metric, fold]
score = np.zeros(shape=(len(clfs), len(metrics), n_folds)) 

# First we need to test how accurate each classificator is on its own

for fold, (train_index, test_index) in enumerate(rsfk.split(X=X, y=y)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    for clf_index, clf in enumerate(clfs):
        clf.fit(X=X_train, y=y_train)
        y_pred = clf.predict(X_test)

        for metric_index, metric in enumerate(metrics): 
            if metric == f1_score:
                metric_score = metric(y_true=y_test, y_pred=y_pred, average='macro')
            else:
                metric_score = metric(y_true=y_test, y_pred=y_pred)
            score[clf_index][metric_index][fold] = metric_score


headers = ["", "GNB", "KNN", "DT"]
rows = [
    ["AC", f"{score[0][0][:].mean():.3} +- {score[0][0][:].std():.3}",  f"{score[1][0][:].mean():.3} +- {score[1][0][:].std():.3}",  f"{score[2][0][:].mean():.3} +- {score[2][0][:].std():.3}"],
    ["BAC", f"{score[0][1][:].mean():.3} +- {score[0][1][:].std():.3}",  f"{score[1][1][:].mean():.3} +- {score[1][1][:].std():.3}",  f"{score[2][1][:].mean():.3} +- {score[2][1][:].std():.3}"],
    ["F1-SCORE", f"{score[0][2][:].mean():.3} +- {score[0][2][:].std():.3}",  f"{score[1][2][:].mean():.3} +- {score[1][2][:].std():.3}",  f"{score[2][2][:].mean():.3} +- {score[2][2][:].std():.3}"],
]

print(tabulate(rows, headers=headers, floatfmt=".3f"))