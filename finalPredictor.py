from sys import version_info
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc
from scipy import interp
from pandas_ml import ConfusionMatrix
import sys
import random
import encoding
import numpy as np
import matplotlib.pyplot as plt


def file_check(filename):
    try:
        file = open(filename, 'r')
        return file
    except IOError:
        print("Error: The file %s seems to NOT exist. Please make sure the file exists in your directory."%(filename))
        sys.exit(0)
        return 0

def read_file(filename):
    file = file_check(filename)
    lines = list(file)
    File = []
    for element in lines:
        element = element.replace('\r','')
        element = element.replace('\n','')
        element = element.split('\t')
        File.append(element)
    return File

def balanceFile(file):
    Dataset = []
    Positive = []
    positive = 0
    Negative = []
    for i in range(len(file)):
        if file[i][2] == '1':
            Positive.append(file[i]); positive += 1
        else:
            Negative.append(file[i]);
    Dataset.extend(Positive)
    Dataset.extend(Negative[0:positive])
    random.shuffle(Dataset)
    return file

def trainClassifier(training, trainingTargets):
    Peptides = np.asarray(training)
    Targets = np.asarray(trainingTargets)
    Peptides, Targets = Peptides[Targets != 2], Targets[Targets != 2]
    n_samples, n_features = Peptides.shape
    # Classification and ROC analysis
    # Run classifier with cross-validation and plot ROC curves
    cv = StratifiedKFold(n_splits=6)
    # GradientBoostingClassifier with optimal parameters
    classifier = GradientBoostingClassifier(n_estimators=140, learning_rate=0.07, min_samples_split=5, min_samples_leaf=20, max_depth=6, max_features=11, subsample=0.5, random_state=10)
    # Train classifier with crossvalidation
    positiveCount = 0
    Weights = []
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    i = 0
    for train, test in cv.split(Peptides, Targets):
        for x in range(len(Targets[train])):
            if (Targets[train][x]) == 1: positiveCount += 1
        for x in range(len(Targets[train])):
            if (Targets[train][x] == 1):
                Weights.append(1 * ((len(Targets[train]) / (2 * positiveCount))))
            else:
                Weights.append(1 * (len(Targets[train]) / (2 * (len(Targets[train]) - positiveCount))))
        classifier.fit(Peptides[train], Targets[train], sample_weight=Weights)
        Weights = []
        positiveCount = 0

    return classifier

def train():
    File = read_file('project_training.txt')
    # delete header row
    File.pop(0)
    # shuffle the dataset
    random.shuffle(File)
    #encode all peptides
    Dataset = encoding.encodeFile(File)
    #balance the dataset to have 50% of each class
    balancedDataset = balanceFile(Dataset)
    #split the balanced Dataset into Training, TrainingTargets, Validation and ValidationTargets
    Training = []
    TrainingTargets = []
    for x in range(len(Dataset)):
        Training.append(Dataset[x][0]); TrainingTargets.append(int(Dataset[x][2]))

    classifier = trainClassifier(Training, TrainingTargets)
    return classifier

def predictPeptide(classifier, inputFile):
    File = read_file(inputFile)
    Dataset = encoding.encodeInputFile(File)
    Peptides = []
    for i in range(len(Dataset)):
        Peptides.append(Dataset[i])
    prediction = classifier.predict(Peptides)
    return prediction

def writeOutputfile(outputFile, inputFile, prediction):
    inputFile = read_file(inputFile)
    file = open(outputFile, "w")
    file.write("%s\t%s\n"%('Id','Prediction1'))
    for e in range(len(inputFile)):
        file.write("%s\t%i\n"%(inputFile[0::][e][0],int(prediction[e])))
    file.close()

def main():
    py3 = version_info[0] > 2 #creates boolean value for test that Python major version > 2

    if py3:
        classifier = train()
        inputFile = ""
        inputFile = input("Please enter the file name containing the peptides you want to predict: ")
        file_check(inputFile)
        prediction = predictPeptide(classifier, inputFile)
        outputFile = input("Please enter a file name in which the prediction should be stored: ")
        writeOutputfile(outputFile, inputFile, prediction)
    else:
        print("Error: Please install python3 to use this modell. \nInformation on how to install python3 can be found under: \n https://docs.python.org/3/using/index.html")
        sys.exit(0)


if __name__ == '__main__': main()
