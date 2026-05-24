import  numpy                       as np
from    imblearn.over_sampling      import SMOTE
from    scipy.stats                 import wilcoxon
from    tabulate                    import tabulate
from    sklearn.naive_bayes         import GaussianNB
from    sklearn.preprocessing       import StandardScaler
from    imblearn.over_sampling      import RandomOverSampler
from    imblearn.under_sampling     import RandomUnderSampler
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
y_preds = [[[], [], []], [[], [], []], [[], [], []]]
y_tests = [[[], [], []], [[], [], []], [[], [], []]]
alpha = 0.05
significant = False
sampling_metods_names = ['ROS', 'RUS', 'SMOTE']
sampling_metods = [RandomOverSampler(), RandomUnderSampler(), SMOTE()]
# Score[classificator, metric, sampling_metod, fold]
score = np.zeros(shape=(len(clfs), len(metrics), len(sampling_metods), n_folds)) 

# Test with data normalization - utilizing StandardScaler
scaler = StandardScaler()

for fold, (train_index, test_index) in enumerate(rsfk.split(X=X, y=y)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    scaler.fit(X=X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for clf_index, clf in enumerate(clfs):
        for sampling_metod_index, sampling_metod in enumerate(sampling_metods):
            sampling_metod = sampling_metods[sampling_metod_index]
            X_train_sampled, y_train_sampled = sampling_metod.fit_resample(X_train_scaled, y_train)
            clf.fit(X=X_train_sampled, y=y_train_sampled)
            y_pred = clf.predict(X_test_scaled)

            for metric_index, metric in enumerate(metrics): 
                if metric == f1_score:
                    metric_score = metric(y_true=y_test, y_pred=y_pred, average='macro')
                else:
                    metric_score = metric(y_true=y_test, y_pred=y_pred)
                score[clf_index][metric_index][sampling_metod_index][fold] = metric_score
    
            y_preds[clf_index][sampling_metod_index].extend(y_pred)
            y_tests[clf_index][sampling_metod_index].extend(y_test)

for id, sampling_metod in enumerate(sampling_metods_names):
    print(f"Sampling metod: {sampling_metod}")
    headers = ["", "GNB", "KNN", "DT"]
    rows = [
        ["AC", f"{score[0][0][id][:].mean():.3} +- {score[0][0][id][:].std():.3}",  f"{score[1][0][id][:].mean():.3} +- {score[1][0][id][:].std():.3}",  f"{score[2][0][id][:].mean():.3} +- {score[2][0][id][:].std():.3}"],
        ["BAC", f"{score[0][1][id][:].mean():.3} +- {score[0][1][id][:].std():.3}",  f"{score[1][1][id][:].mean():.3} +- {score[1][1][id][:].std():.3}",  f"{score[2][1][id][:].mean():.3} +- {score[2][1][id][:].std():.3}"],
        ["F1-SCORE", f"{score[0][2][id][:].mean():.3} +- {score[0][2][id][:].std():.3}",  f"{score[1][2][id][:].mean():.3} +- {score[1][2][id][:].std():.3}",  f"{score[2][2][id][:].mean():.3} +- {score[2][2][id][:].std():.3}"],
    ]
    print(tabulate(rows, headers=headers, floatfmt=".3f"))

    print("-----------------------------------------------------------")

    for clf_index_1, clf_1 in enumerate(clfs):
        for clf_index_2, clf_2 in enumerate(clfs):
            if clf_index_1 < clf_index_2:
                # Metric used in wilcoxon test is F1-score
                wilcoxon_test = wilcoxon(score[clf_index_1][2][id][:], score[clf_index_2][2][id][:])
                if wilcoxon_test.pvalue < alpha:
                    significant = ''
                else:
                    significant = 'not '
                print(f"Difference between {clfs_names[clf_index_1]} and {clfs_names[clf_index_2]} is {significant}statistically significant.")

    print("-----------------------------------------------------------")

for clf_index in range(len(clfs)):
    for sampling_index in range(len(sampling_metods)):
        print(f"Confusion matrix {clfs_names[clf_index]} - {sampling_metods_names[sampling_index]}:")
        print(confusion_matrix(y_true=y_tests[clf_index][sampling_index], y_pred=y_preds[clf_index][sampling_index], normalize='true'))