import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

import matplotlib
matplotlib.use('Agg')

# read dataset
dataset = pd.read_csv('./data/heart.csv')
# dataset.info()

# split dataset into train and test sets
array = dataset.values
X = array[:, 0:-1]
Y = array[:, -1]

# print(X[:10])
# print(Y[:10])


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10)


# function to calculate sensitivity and specificity
def sensitivityAndSpecificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp+fn)
    specificity = tn / (tn+fp)
    return sensitivity, specificity

# dataframes for presenting results
trainingResults = pd.DataFrame(index = ['sensitivity', 'specificity'])
testResults = pd.DataFrame(index = ['sensitivity', 'specificity'])

def addToTrainingD(method):
    sensitivity, specificity = sensitivityAndSpecificity(y_train, method.predict(x_train))
    trainingResults['Decision Tree'] = np.array([sensitivity, specificity], dtype=np.float32)

# function for plotting convergence for Random Forest with increasing number of parameters
def plotRFconvergence():
    n = 60
    sensitivity = np.zeros(n)
    specificity = np.zeros(n)
    x = np.linspace(1, n, n)
    for i in range(n):
        RandomForest = RandomForestClassifier(n_estimators=i+1)
        RandomForest.fit(x_train, y_train)
        sensitivity[i], specificity[i] = sensitivityAndSpecificity(
            y_test, RandomForest.predict(x_test))
    fig = plt.figure()
    plt.plot(x, sensitivity, label='sensitivity')
    plt.plot(x, specificity, label='specificity')
    plt.title('Metrics for Random Forest with increasing number of estimators')
    axes = plt.gca()
    axes.set_ylim([0, 1])
    axes.set_xlim([1, n+1])
    plt.xlabel('number of estimators')
    plt.ylabel('score')
    plt.legend()
    fig.savefig('RFconvergence.png')


plotRFconvergence()

# ------------------------- Decition Tree (To see difference) -----------------------------------------
dTree = DecisionTreeClassifier()

dTree.fit(x_train, y_train)

sensitivity, specificity = sensitivityAndSpecificity(
    y_train, dTree.predict(x_train))
trainingResults['Decision Tree'] = np.array(
    [sensitivity, specificity], dtype=np.float32)

sensitivity, specificity = sensitivityAndSpecificity(
    y_test, dTree.predict(x_test))
testResults['Decision Tree'] = np.array(
    [sensitivity, specificity], dtype=np.float32)

fpr_dt, tpr_dt, _ = roc_curve(y_test, dTree.predict_proba(x_test)[:, 1])



# ---------------------------- Random Forest Classifier (Bagging) -------------------------------------------

RandomForest = RandomForestClassifier(n_estimators = 50, max_samples = 0.5)
RandomForest.fit(x_train,y_train)

sensitivity, specificity = sensitivityAndSpecificity(
    y_train, RandomForest.predict(x_train))
trainingResults['Random Forest'] = np.array(
    [sensitivity, specificity], dtype=np.float32)

sensitivity, specificity = sensitivityAndSpecificity(
    y_test, RandomForest.predict(x_test))
testResults['Random Forest'] = np.array(
    [sensitivity, specificity], dtype=np.float32)

fpr_rf, tpr_rf, _ = roc_curve(y_test, RandomForest.predict_proba(x_test)[:, 1])

# ------------------------- Bagging (w/ decision trees) -------------------------------------------------

Bagging = BaggingClassifier(DecisionTreeClassifier(), max_samples=0.5, max_features=1.0, n_estimators=50)
# NOTE: bad results. needs tuning
Bagging.fit(x_train, y_train)

sensitivity, specificity = sensitivityAndSpecificity(
    y_train, Bagging.predict(x_train))
trainingResults['Bagging'] = np.array(
    [sensitivity, specificity], dtype=np.float32)

sensitivity, specificity = sensitivityAndSpecificity(
    y_test, Bagging.predict(x_test))
testResults['Bagging'] = np.array([sensitivity, specificity], dtype=np.float32)

fpr_b, tpr_b, _ = roc_curve(y_test, Bagging.predict_proba(x_test)[:, 1])

# AdaBoost (Boosting) ----------------------------------------------------------

ada = AdaBoostClassifier(n_estimators=60, learning_rate=1)
ada.fit(x_train, y_train)

sensitivity, specificity = sensitivityAndSpecificity(
    y_train, ada.predict(x_train))
trainingResults['AdaBoost'] = np.array(
    [sensitivity, specificity], dtype=np.float32)

sensitivity, specificity = sensitivityAndSpecificity(
    y_test, ada.predict(x_test))
testResults['AdaBoost'] = np.array(
    [sensitivity, specificity], dtype=np.float32)

fpr_ab, tpr_ab, _ = roc_curve(y_test, ada.predict_proba(x_test)[:, 1])


# ----------------------------- Voting --------------------------------------------
"""
lr = LogisticRegression(max_iter=500)
dt = DecisionTreeClassifier()
svm = SVC(kernel = 'poly', degree = 2 )
kn = KNeighborsClassifier(n_neighbors=7)
clf = MultinomialNB()

Voting = VotingClassifier( estimators= [('clf',clf),('dt',dt),('svm',svm), ('lr', lr), ('kn', kn)], voting = 'hard')

Voting.fit(x_train, y_train)

print("-- VOTING --")

sensitivity, specificity = sensitivityAndSpecificity(y_train, Voting.predict(x_train))
trainingResults['Voting'] = np.array([sensitivity, specificity], dtype=np.float32)

sensitivity, specificity = sensitivityAndSpecificity(y_test, Voting.predict(x_test))
testResults['Voting'] = np.array([sensitivity, specificity], dtype=np.float32)
"""

# run functions ------------------

# Formate and print results 
trainingResults = trainingResults.T
testResults = testResults.T

print('\n --Results on training data--')
print(trainingResults)

print('\n --Results on test data--')
print(testResults)

def scatterPlot():
    # plot results in scatter plot
    # must have four methods
    fig, ax = plt.subplots()
    ax.scatter(testResults.sensitivity.values, testResults.specificity.values, c=[
'tab:blue', 'tab:orange', 'tab:green', 'tab:red'])  # add  'tab:gray' for voting
    ax.set_xlim((0.5, 1))
    ax.set_ylim((0.5, 1))
    plt.xlabel('sensitivity')
    plt.ylabel('specificity')
    # ax.legend()
    ax.grid(True)

    for i in range(len(testResults.index)):
        ax.annotate(
            testResults.index[i], (testResults.sensitivity.values[i], testResults.specificity.values[i]))

    fig.savefig('data/test.png')

def rocPlot():
    fig = plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_dt, tpr_dt, label='DT (area = %0.2f)' % auc(fpr_dt, tpr_dt))
    plt.plot(fpr_rf, tpr_rf, label='RF(area = %0.2f)' % auc(fpr_rf, tpr_rf))
    plt.plot(fpr_b, tpr_b, label='Bagging (area = %0.2f)' % auc(fpr_b, tpr_b))
    plt.plot(fpr_ab, tpr_ab, label='AdaBoost(area = %0.2f)' % auc(fpr_ab, tpr_ab))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    fig.savefig('data/rocplot.png')

rocPlot()


