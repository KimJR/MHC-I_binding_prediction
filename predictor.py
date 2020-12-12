from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.utils.class_weight import compute_sample_weight
from pandas_ml import ConfusionMatrix
import matplotlib.pyplot as plt

import random
import numpy as np
import random
import argparse
from sklearn.model_selection import train_test_split
import sklearn.metrics

def A():
    return [-0.57, 0.39, -0.96, -0.61, -0.69]

def R():
    return [-0.40, -0.83, -0.61, 1.26, -0.28]

def N():
    return [-0.70, -0.63, -1.47, 1.02, 1.06]

def D():
    return [-1.62, -0.52, -0.67, 1.02, 1.47]

def C():
    return [0.07, 2.04, 0.65, -1.13, -0.39]

def Q():
    return [-0.05, -1.50, -0.67, 0.49, 0.21]

def E():
    return [-0.64, -1.59, -0.39, 0.69, 1.04]

def G():
    return [-0.90, 0.87, -0.36, 1.08, 1.95]

def H():
    return [0.73, -0.67, -0.42, 1.13, 0.99]

def I():
    return [0.59, 0.79, 1.44, -1.90, -0.93]

def L():
    return [0.65, 0.84, 1.25, -0.99, -1.90]

def K():
    return [-0.64, -1.19, -0.65, 0.68, -0.13]

def M():
    return [0.76, 0.05, 0.06, -0.62, -1.59]

def F():
    return [1.87, 1.04, 1.28, -0.61, -0.16]

def P():
    return [-1.82, -0.63, 0.32, 0.03, 0.68]

def S():
    return [-0.39, -0.27, -1.51, -0.25, 0.31]

def T():
    return [-0.04, -0.30, -0.82, -1.02, -0.04]

def W():
    return [1.38, 1.69, 1.91, 1.07, -0.05]

def Y():
    return [1.75, 0.11, 0.65, 0.21, -0.41]

def V():
    return [-0.02, 0.30, 0.97, -1.55, -1.16]

def BLOMAP(char):
    switcher = {
        'A': A(),
        'R': R(),
        'N': N(),
        'D': D(),
        'C': C(),
        'Q': Q(),
        'E': E(),
        'G': G(),
        'H': H(),
        'I': I(),
        'L': L(),
        'K': K(),
        'M': M(),
        'F': F(),
        'P': P(),
        'S': S(),
        'T': T(),
        'W': W(),
        'Y': Y(),
        'V': V(),
    }
    func = switcher.get(char, lambda: "Char is not a amino acid")
    return func


Training = []
TrainingTargets = []
Validation = []
ValidationTargets = []

def train():
    filename = 'project_training.txt'
    file = open(filename, 'r')
    lines = list(file)
    File = []
    Peptides = []
    Targets = []

    for element in lines:
        element = element.replace('\r', '')
        element = element.replace('\n', '')
        element = element.split('\t')
        File.append(element)

    # delete header row
    File.pop(0)
    # shuffle the dataset
    random.shuffle(File)
    for element in enumerate(File[0::]):
        for peptide in element[1][0:1:]:
            ascii = []
            for char in peptide:
                encoded = BLOMAP(char)
                ascii.append(encoded[0])
                ascii.append(encoded[1])
                ascii.append(encoded[2])
                ascii.append(encoded[3])
                ascii.append(encoded[4])
            File[element[0]][0] = ascii
    Dataset = []
    Positive = []
    positive = 0
    Negative = []
    negative = 0

    for i in range(len(File)):
        if File[i][2] == '1':
            Positive.append(File[i]); positive += 1
        else:
            Negative.append(File[i]); negative +=1
    Dataset.extend(Positive)
    Dataset.extend(Negative[0:positive])
    #Dataset.extend(File)
    random.shuffle(Dataset)

    Training = []
    TrainingTargets = []
    Validation = []
    ValidationTargets = []
    #for x in range(positive):
    #    if random.uniform(0,1) >= 0.2:
    #        Training.append(Positive[x][0]);
    #        TrainingTargets.append(int(Positive[x][2]))
    #        Training.append(Negative[x][0]);
    #        TrainingTargets.append(int(Negative[x][2]))
    #    else:
    #        Validation.append(Positive[x][0]);
    #        ValidationTargets.append(int(Positive[x][2]))
    #        Validation.append(Negative[x][0]);
    #        ValidationTargets.append(int(Negative[x][2]))


    for x in range(len(Dataset)):
        if random.uniform(0, 1) >= 0.0: #Steht bewusst auf 0 damit bei der Cross-Validation keine Information Verloren geht
            Training.append(Dataset[x][0]); TrainingTargets.append(int(Dataset[x][2]))
        else:
            Validation.append(Dataset[x][0]); ValidationTargets.append(int(Dataset[x][2]))
                     
    #Training = []
    #TrainingTargets = []
    #Validation = []
    #ValidationTargets = []
    #for x in range(len(File)):
    #    if random.uniform(0, 1) >= 0.2: #Hier keine 0 damit wir den Fit durchfuehren und pruefen koennen
    #        Training.append(File[x][0]); TrainingTargets.append(int(File[x][2]))
    #    else:
    #        Validation.append(File[x][0]); ValidationTargets.append(int(File[x][2]))

    #param_test1 = {}
    #gsearch1 = GridSearchCV(
    #        estimator=GradientBoostingClassifier(n_estimators=140, learning_rate=0.07, min_samples_split=5, min_samples_leaf=20, max_depth=6, max_features=34, subsample=0.5, random_state=10),
    #
    #
    #                            param_grid = param_test1, scoring = 'roc_auc', n_jobs = 4, iid = False, cv = 6)
    #lb = preprocessing.LabelBinarizer()
    #TrainingTargets = np.array([number[0] for number in lb.fit_transform(TrainingTargets)])
    #gsearch1.fit(Training, TrainingTargets)
    #lb = preprocessing.LabelBinarizer()
    #TrainingTargets = np.array([number[0] for number in lb.fit_transform(TrainingTargets)])
    #print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)

    #ValidationPrediction = gsearch1.predict(Validation)
    #cm = ConfusionMatrix(ValidationTargets, ValidationPrediction)
    #print(cm)
    #cm.plot()
    #plt.show()
    #cm.print_stats()




    import numpy as np
    from scipy import interp
    import matplotlib.pyplot as plt
    from itertools import cycle

    from sklearn import svm, datasets
    from sklearn.metrics import roc_curve, auc
    from sklearn.model_selection import StratifiedKFold

    # #############################################################################
    # Data IO and generation

    # Import some data to play with
    iris = datasets.load_iris()
    X = np.asarray(Training)
    y = np.asarray(TrainingTargets)
    X, y = X[y != 2], y[y != 2]
    n_samples, n_features = X.shape

    # #############################################################################
    # Classification and ROC analysis

    # Run classifier with cross-validation and plot ROC curves
    cv = StratifiedKFold(n_splits=6)
    #1
    #classifier = GradientBoostingClassifier(n_estimators=73, learning_rate=0.072, min_samples_split=70, min_samples_leaf=50, max_depth=4, subsample=0.75)
    #2
    #classifier = GradientBoostingClassifier(n_estimators=73, learning_rate=0.072, min_samples_split=70, min_samples_leaf=50, max_depth=4, subsample=0.5)
    #3
    #classifier = GradientBoostingClassifier(n_estimators=73, learning_rate=0.072, min_samples_split=70, min_samples_leaf=50, max_depth=4, subsample=1, max_features=7)
    #4 (optimal parameters from grid search)
    #classifier = GradientBoostingClassifier(learning_rate=0.075, n_estimators=70, max_depth=4, min_samples_split=40, min_samples_leaf=9, max_features=6 , subsample=0.8)
    #5 top
    classifier = GradientBoostingClassifier(n_estimators=140, learning_rate=0.07, min_samples_split=5, min_samples_leaf=20, max_depth=6, max_features=11, subsample=0.5, random_state=10)
    #classifier = svm.SVC(kernel='linear', probability=True, random_state=random_state)

    
##################Das#################################    
#    poscount = 0
#    Weights = []
#    i = 0
#    for train, test in cv.split(X, y):
#
#        for x in range(len(y[train])):
#            if (y[train][x]) == 1: poscount += 1
#
#        for x in range(len(y[train])):
#            if (y[train][x] == 1):
#                Weights.append(1 * ((len(y[train]) / (2 * poscount))))
#            else:
#                Weights.append(1 * (len(y[train]) / (2 * (len(y[train]) - poscount))))
#
#        print(Weights)
#        print(y[train], len(y[train]), poscount)
#        probas_ = classifier.fit(X[train], y[train],sample_weight= Weights).predict_proba(
#            X[test])  # probas_ = classifier.fit(X[train], y[train],sample_weight=Weights).predict_proba(X[test])
#        # Compute ROC curve and area the curve
#        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
#        tprs.append(interp(mean_fpr, fpr, tpr))
#        tprs[-1][0] = 0.0
#        roc_auc = auc(fpr, tpr)
#        aucs.append(roc_auc)
#        plt.plot(fpr, tpr, lw=1, alpha=0.3,
#                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
#
#        i += 1
#        Weights=[]
#        poscount=0
    
    
    
    
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    i = 0
    for train, test in cv.split(X, y):
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test]) #probas_ = classifier.fit(X[train], y[train],sample_weight=Weights).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        i += 1
        #muss hier hin###################################
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return classifier

    #ValidationPrediction = classifier.predict(Validation)
    #cm = ConfusionMatrix(ValidationTargets, ValidationPrediction)
    #print(cm)
    #cm.plot()
    #plt.show()
    #cm.print_stats()

    #clf = svm.SVC(C=1.0, kernel='rbf')
    #clf = GradientBoostingClassifier(n_estimators=73, learning_rate=0.072, min_samples_split=70, min_samples_leaf=50, max_depth=4, subsample=0.75)
    #clf.fit(Training, TrainingTargets)
    #ValidationPrediction = clf.predict(Validation)
    #print(ValidationPrediction)
    #cm = ConfusionMatrix(ValidationTargets, ValidationPrediction)
    #print(cm)
    #cm.plot()
    #plt.show()
    #cm.print_stats()

parser = argparse.ArgumentParser(description = "Random Epitope Binding Predictor")
parser.add_argument('input', metavar='in-file')
parser.add_argument('output', metavar='out-file')
args = parser.parse_args()
with open(args.output, "w") as o:
    with open (args.input, "r") as i:
        clf = train()
        lines = list(i)
        File = []
        Peptides = []
        for element in lines:
            element = element.replace('\r','')
            element = element.replace('\n','')
            element = element.split(',')
            File.append(element)
        header = File.pop(0)
        for element in enumerate(File[0::]):
            for peptide in element[1][0:1:]:
                ascii = []
                for char in peptide:
                    encoded = BLOMAP(char)
                    ascii.append(encoded[0])
                    ascii.append(encoded[1])
                    ascii.append(encoded[2])
                    ascii.append(encoded[3])
                    ascii.append(encoded[4])
                Peptides.append(ascii)
        prediction = clf.predict(Peptides)
        o.write("%s,%s\n"%('Id','Prediction1'))
        for e in range(len(File)):
            o.write("%s,%i\n"%(File[0::][e][0],int(prediction[e])))
