import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

#read dataset
dataset = pd.read_csv('./data/heart.csv')
#dataset.info()

#split dataset into train and test sets
array = dataset.values
X = array[:,0:-1]
Y = array[:,-1]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=4)

# Decition Tree (To see difference)

dTree = DecisionTreeClassifier()

dTree.fit(x_train,y_train)

print("Decision Tree on test data",dTree.score(x_test, y_test))

print("Decision Tree on training data",dTree.score(x_train, y_train))

#Random Forest Classifier

RandomForest = RandomForestClassifier(n_estimators = 30)
RandomForest.fit(x_train,y_train)

print("Random Forest on test data",RandomForest.score(x_test, y_test))

print("Random Forest on training data",RandomForest.score(x_train, y_train))

