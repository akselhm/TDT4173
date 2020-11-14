import matplotlib.pyplot as plt
# import main
import pandas as pd
import seaborn as sns

#--------- Extracting data from the data set ----------#
dataset = pd.read_csv('./data/heart.csv')
genderData = dataset["sex"]
ageData = dataset["age"]

# -------- Representing genders in study --------#
maleCounter = 0
femaleCounter = 0
fig, ax = plt.subplots()

for i in genderData:
    if i == 1:
        maleCounter += 1
    else:
        femaleCounter += 1

gender = ["Male", "Female"]
participants = [maleCounter, femaleCounter]
newColors = ['lightblue', 'red']
plt.bar(gender, participants, color=newColors)
plt.title('Male and Female participants', fontsize=14)
plt.xlabel('Gender', fontsize=14)
plt.ylabel('Number of individuals', fontsize=14)
plt.grid(False)
fig.savefig('data/genders.png')

#-------- Representing ages in study --------#

age = dataset['age'].value_counts()


#-------- Representing heart disease in study --------#
fig1, ax = plt.subplots()
dataset.target.value_counts().plot(
    kind="bar", color=["#F74441", "#8DF790"])
fig1.savefig('data/heartDisease.png')


#---------- Representing with a correlation matrix ----------#

corrMatrix = dataset.corr()
fig3, ax = plt.subplots(figsize=(15, 15))
ax = sns.heatmap(corrMatrix, annot=True, linewidth=0.5,
                 fmt=".2f", cmap="YlOrRd")
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
fig3.savefig('data/corrMatrix.png')


#----------- Representing attributes in corr with the target ------------#

fig4, ax = plt.subplots()
dataset.drop('target', axis=1).corrwith(dataset.target).plot(
    kind='barh', grid=True, figsize=(12, 8), title="Correlations with heart disease", color=["#6E0202"])
fig4.savefig('data/corrWithHeartDisease.png')


#------------- Histograph of ages -------------#

fig5, ax = plt.subplots()
dataset["age"].plot(kind="hist", grid=True, figsize=(12, 8), title="Ages")
fig5.savefig('data/histOfAges.png')


#--------- Histographs of attributes and corr with target ----------#


categoricalValues = []
continousValues = []

for column in dataset.columns:
    if len(dataset[column].unique()) <= 10:
        categoricalValues.append(column)
    else:
        continousValues.append(column)

fig6, ax = plt.subplots(figsize=(15, 15))
for i, column in enumerate(categoricalValues, 1):
    plt.subplot(3, 3, i)
    dataset[dataset['target'] == 0][column].hist(
        bins=35, color='green', label="No heart disease", alpha=0.6)
    dataset[dataset['target'] == 1][column].hist(
        bins=35, color='red', label="Heart disease", alpha=0.6)
    plt.legend()
    plt.xlabel(column)
fig6.savefig('data/hists.png')

fig7, ax = plt.subplots(figsize=(15, 15))
for i, column in enumerate(continousValues, 1):
    plt.subplot(3, 3, i)
    dataset[dataset['target'] == 0][column].hist(
        bins=35, color='green', label="No heart disease", alpha=0.6)
    dataset[dataset['target'] == 1][column].hist(
        bins=35, color='red', label="Heart disease", alpha=0.6)
    plt.legend()
    plt.xlabel(column)
fig7.savefig('data/hists2.png')
