
# Heart Disease Prediction Using Ensamble Methods

This repository contains the program of a graded project in the course TDT4173 - Machine Learning at Norwegian University of Science and Technology.


### NOTE ###

The code is heavily commented and should be intutive for the novel reader.

## Motivation
This project was motivated by the rapid integration of machine learning methods in the field of medicine. Since heart diseases rarely show any symptoms programs, such as ours, can have an enourmous impact on the speed and reliability of diaganosis and prognosis on patients who may have heart diseases. 


## Language and libraries

This program uses python 3.8.1 and the following libraries:

         * Scikit-Learn (sk-learn)
         * Numpy
         * Seaborn
         * Matplotlib
         * Pandas
         


## Methods

The Ensamble Methods used in the program are the following:

         * ### AdaBoost ###
         https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
         * ### RandomForest ###
         https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
         * ### Decision Trees ###
         https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

Their respective documentation on the Scikit-Learn website is provided under each method. 

## Running the program

Depending on what editor you use the initiation of the program will differ. The editor used in the project was Visual Studio Code and as such running the program became extensivly easy with the plug in: "Code Runner".
To add this extension in VScode go to the marketplace, type in "code runner" and install:



<img width="303" alt="Screenshot 2020-11-26 at 21 19 30" src="https://user-images.githubusercontent.com/43234635/100390542-0d52e800-3031-11eb-86ea-a437bf18573e.png">


After installation, to run the program simply press the green "play" button in the top righ corner of the header.

If this is not an option for you can also use the terminal in your prefered editor to run the program. 
It is recommended to use a virtual environment when you run from the terminal.
A virtual environment (venv) with the required libraries can be made following this guide: https://code.visualstudio.com/docs/python/data-science-tutorial

Once your venv is up and running type "python main.py" or "python3 main.py" (depnding on which version of python you have in your path) to run the program. The results wil print in the terminal.
NB! Make sure you have navigated to the folder where the program is located.

Run "python data_representation.py" or "python3 data_representation.py" for visuals of the dataset. These will appear in the data folder.

Estimated run time of the program is: 18 sec
