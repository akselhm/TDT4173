import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, make_scorer, accuracy_score, recall_score, plot_roc_curve
import seaborn as sns
import matplotlib
matplotlib.use('Agg')

# read dataset
dataset = pd.read_csv('./data/heart.csv')

# split dataset
array = dataset.values
X = array[:, 0:-1]
Y = array[:, -1]

# make Statisfied K-Fold Crossvalidation
kfold = StratifiedKFold(n_splits=5)

# Define metrics to evaluate  
metrics = {'accuracy': make_scorer(accuracy_score),
    'sensitivity': make_scorer(recall_score),
    'specificity': make_scorer(recall_score,pos_label=0.0)}


# dataframe for saving and presenting results
testResults = pd.DataFrame(index = ['accuracy','sensitivity', 'specificity'])


def calcMetrics(classifier):
    # function for calculating the metric for a classifier using K-Fold Cross Validation
    # Metrics calculated are accuracy, sensitivity and specificity (given in metrics dictionary)
    # returns: a list with the mean values for the metrics [accuracy, sensitivity, specificity]

    results = cross_validate(classifier, X, Y, cv=kfold, scoring= metrics)
    print(results) # print results on test set for all folds, as well as training time and score time

    acc = np.mean(results.get('test_accuracy'))
    sens = np.mean(results.get('test_sensitivity'))
    spec = np.mean(results.get('test_specificity'))
    return [acc, sens, spec]


def rocCurveKFold(classifier, name):
    # function for plotting a roc curve for a given classifier
    # A roc curve is produced for each fold in the kFold, as well as a mean ROC
    # returns: mean false positive rate and false negative rate
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig1, ax = plt.subplots()
    for i, (train, test) in enumerate(kfold.split(X, Y)):
        classifier.fit(X[train], Y[train])
        viz = plot_roc_curve(classifier, X[test], Y[test],
                            name='ROC fold {}'.format(i),
                            alpha=0.3, lw=1, ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
        title="Receiver operating characteristic " + name)
    ax.legend(loc="lower right")
    fig1.savefig('results/rocplot'+name+'.png')

    return mean_fpr, mean_tpr

# ------------------ dTree ------------------------

# make classifier
dTree = DecisionTreeClassifier()

# calculate metrics for Decision Tree and add to dataframe
dTreemetrics = calcMetrics(dTree)
testResults['Decision Tree'] = np.array(
    dTreemetrics, dtype=np.float32)

# make an array with predicted values for the descision tree
# the array is used to make a confusion matrix
predDT = cross_val_predict(dTree, X, Y, cv=kfold)

# plot a roc curve for the decision tree
# save the  mean true positive rate and false negative rate
fpr_dt, tpr_dt = rocCurveKFold(dTree, 'dTree')

# ------------------- RF --------------------------

rf = RandomForestClassifier(n_estimators = 50, max_samples=0.5)

rfmetrics = calcMetrics(rf)
testResults['Random Forest'] = np.array(
    rfmetrics, dtype=np.float32)

predRF = cross_val_predict(rf, X, Y, cv=kfold)

fpr_rf, tpr_rf = rocCurveKFold(rf, 'RF')

# ----------------- AdaBoost ----------------------

ada = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=4), n_estimators = 100, learning_rate=1, algorithm= 'SAMME' )

adametrics = calcMetrics(ada)
testResults['AdaBoost'] = np.array(
    adametrics, dtype=np.float32)

predAda = cross_val_predict(ada, X, Y, cv=kfold)
fpr_ab, tpr_ab = rocCurveKFold(ada, 'AB')


# run functions ------------------

# Formate and print results 
testResults = testResults.T

print('\n --Results on test data--')
print(testResults)

def scatterPlot():
    # plot results in scatter plot
    fig, ax = plt.subplots()
    ax.scatter(testResults.sensitivity.values, testResults.specificity.values)  
    ax.set_xlim((0.5, 1))
    ax.set_ylim((0.5, 1))
    plt.xlabel('sensitivity')
    plt.ylabel('specificity')
    # ax.legend()
    ax.grid(True)

    for i in range(len(testResults.index)):
        ax.annotate(
            testResults.index[i], (testResults.sensitivity.values[i], testResults.specificity.values[i]))

    fig.savefig('results/test.png')

def rocPlot():
    fig = plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_dt, tpr_dt, label='DT (area = %0.2f)' % auc(fpr_dt, tpr_dt))
    plt.plot(fpr_rf, tpr_rf, label='RF(area = %0.2f)' % auc(fpr_rf, tpr_rf))
    #plt.plot(fpr_b, tpr_b, label='Bagging (area = %0.2f)' % auc(fpr_b, tpr_b))
    plt.plot(fpr_ab, tpr_ab, label='AdaBoost(area = %0.2f)' % auc(fpr_ab, tpr_ab))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    fig.savefig('results/rocplot.png')

rocPlot()


#--------- Making the confusion matrices----------#

def confM(true_y, pred_y):
    # function for making the confusion matrices
    # parameters: 
    #   y_true: 
    # returns: a figure of the confusion matrix
    figmat,ax = plt.subplots()
    data = {'y_Actual':    true_y, 
            'y_Predicted': pred_y }

    df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
   # group_names = ["True Neg","False Pos","False Neg","True Pos"]
    mat = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'], margins = True)

    #text = np.array([['TN', 'FP', 'AN'], ['FN', 'TP', 'AP'], ['PN', 'PP', 'T']])
   # labels = (np.array(["{0}\n{1:.2f}".format(text,data) for text, data in zip(text.flatten(), mat.flatten())])).reshape(3,3)
    sns.set(font_scale=1.6)
    sns.heatmap(mat, annot=True, fmt='', cbar = False) #cbar=False,square=True
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    ax.tick_params( labelsize=15)
    
    return figmat
    #figmat.savefig('data/confusionMatrix.png')
 

figAda = confM(Y, predAda)
figAda.savefig('results/confusionMatrixAda.png')

figDT = confM(Y, predDT)
figDT.savefig('results/confusionMatrixDT.png')

figRF = confM(Y, predRF)
figRF.savefig('results/confusionMatrixRF.png')
