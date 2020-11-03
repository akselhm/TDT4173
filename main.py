import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

#read dataset
dataset = pd.read_csv('./data/heart.csv')
#dataset.info()

#split dataset into train and test sets
array = dataset.values
X = array[:,0:-1]
Y = array[:,-1]

#print(X[:10])
#print(Y[:10])

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=10)

#function to calculate sensitivity and specificity
def sensitivityAndSpecificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp+fn)
    specificity = tn / (tn+fp)
    return sensitivity, specificity

# ------------------------- Decition Tree (To see difference) -----------------------------------------

dTree = DecisionTreeClassifier()

dTree.fit(x_train,y_train)

#print("Decision Tree on test data",dTree.score(x_test, y_test))
#print("Decision Tree on training data",dTree.score(x_train, y_train))

print("-- DECISION TREE --")

sensitivity, specificity = sensitivityAndSpecificity(y_train, dTree.predict(x_train))
print("sensitivity on training data: ", sensitivity)
print("specificity on training data: ", specificity)

sensitivity, specificity = sensitivityAndSpecificity(y_test, dTree.predict(x_test))
print("sensitivity on test data: ", sensitivity)
print("specificity on test data: ", specificity)
print("\n")


#---------------------------- Random Forest Classifier (Bagging) -------------------------------------------

RandomForest = RandomForestClassifier(n_estimators = 30)
RandomForest.fit(x_train,y_train)

#print("Random Forest on test data",RandomForest.score(x_test, y_test))
#print("Random Forest on training data",RandomForest.score(x_train, y_train))

print("-- RANDOM FOREST --")

sensitivity, specificity = sensitivityAndSpecificity(y_train, RandomForest.predict(x_train))
print("sensitivity on training data: ", sensitivity)
print("specificity on training data: ", specificity)

sensitivity, specificity = sensitivityAndSpecificity(y_test, RandomForest.predict(x_test))
print("sensitivity on test data: ", sensitivity)
print("specificity on test data: ", specificity)
print("\n")


#------------------------- Bagging (w/ decision trees) -------------------------------------------------

Bagging = BaggingClassifier(DecisionTreeClassifier(), max_samples= 0.5, max_features = 1.0, n_estimators = 30)
#NOTE: bad results. needs tuning
Bagging.fit(x_train,y_train)

#print("Bagging Forest on test data",Bagging.score(x_test, y_test))
#print("Bagging on training data",Bagging.score(x_train, y_train))

print("-- BAGGING --")

sensitivity, specificity = sensitivityAndSpecificity(y_train, Bagging.predict(x_train))
print("sensitivity on training data: ", sensitivity)
print("specificity on training data: ", specificity)

sensitivity, specificity = sensitivityAndSpecificity(y_test, Bagging.predict(x_test))
print("sensitivity on test data: ", sensitivity)
print("specificity on test data: ", specificity)
print("\n")

# AdaBoost (Boosting) ----------------------------------------------------------




# ----------------------------- Voting --------------------------------------------

lr = LogisticRegression()
dt = DecisionTreeClassifier()
svm = SVC(kernel = 'poly', degree = 2 )

Voting = VotingClassifier( estimators= [('lr',lr),('dt',dt),('svm',svm)], voting = 'hard')

Voting.fit(x_train, y_train)

print("-- VOTING --")

sensitivity, specificity = sensitivityAndSpecificity(y_train, Voting.predict(x_train))
print("sensitivity on training data: ", sensitivity)
print("specificity on training data: ", specificity)

sensitivity, specificity = sensitivityAndSpecificity(y_test, Voting.predict(x_test))
print("sensitivity on test data: ", sensitivity)
print("specificity on test data: ", specificity)
print("\n")