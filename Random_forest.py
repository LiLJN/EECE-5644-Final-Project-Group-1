import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import string
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
all = [train, test]
##show missing value
def show_missing(data):     
    for col in data.columns.tolist():          
        print('{} missing values: {}'.format(col, data[col].isnull().sum()))
    print('\n')

for dataset in all:    
    #complete missing age with median
    dataset['Age'].fillna(dataset['Age'].median(), inplace = True)

    #complete embarked with mode
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)

    #complete missing fare with median
    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)

#delete the cabin feature/column and others previously stated to exclude in train dataset
drop_column = ['PassengerId','Cabin', 'Ticket']
train.drop(drop_column, axis=1, inplace = True)
test.drop(drop_column, axis=1, inplace = True)


print(train.isnull().sum())
print("-"*10)
print(test.isnull().sum())




g = sns.FacetGrid(train, col='Survived')
g.map(sns.distplot, "Age")

# Explore Age distibution 
fig, ax = plt.subplots()
g = sns.kdeplot(train["Age"][(train["Survived"] == 0)], color="Red", shade = True)
g = sns.kdeplot(train["Age"][(train["Survived"] == 1)], ax =g, color="Green", shade= True)
g.set_xlabel("Age")
g.set_ylabel("Frequency")
g = g.legend(["Not Survived","Survived"])

fig, ax = plt.subplots()
pal = {1:"Green", 0:"Red"}
g = sns.FacetGrid(train, col="Sex", row="Survived", margin_titles=True, hue = "Survived", palette=pal)
g = g.map(plt.hist, "Age", edgecolor = 'white')
g.fig.suptitle("Survived by Sex and Age")
plt.subplots_adjust(top=0.90)

fig, ax = plt.subplots()
pal = {1:"Green", 0:"Red"}
g = sns.FacetGrid(train, col="Sex", row="Pclass", margin_titles=True, hue = "Survived",palette = pal)
g = g.map(plt.hist, "Age", edgecolor = 'white').add_legend()
g.fig.suptitle("Survived by Sex，Age and Pclass")
plt.subplots_adjust(top=0.90)

fig, ax = plt.subplots()
pal = {1:"Green", 0:"Red"}
g = sns.FacetGrid(train, col="Sex", row="Embarked", margin_titles=True, hue = "Survived",palette = pal)
g = g.map(plt.hist, "Age", edgecolor = 'white').add_legend()
g.fig.suptitle("Survived by Sex, Age and Embarked")
plt.subplots_adjust(top=0.90)


#map each Embarked value to a numerical value
embarked_mapping = {"S": 1, "C": 2, "Q": 3}
train['Embarked'] = train['Embarked'].map(embarked_mapping)
test['Embarked'] = test['Embarked'].map(embarked_mapping)

sex_mapping = {"male": 1, "female": 0}
train['Sex'] = train['Sex'].map(sex_mapping)
test['Sex'] = test['Sex'].map(sex_mapping)

####model data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
##age

# Random Forest
from sklearn.ensemble import RandomForestClassifier
X1 =  train[["Age", "Sex"]]
Y1 = train["Survived"]
X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1,test_size = .33, random_state=0)
randomforest = RandomForestClassifier()
randomforest.fit(X1_train, Y1_train)
Y1_pred = randomforest.predict(X1_test)
acc1_randomforest = round(accuracy_score(Y1_pred, Y1_test) * 100, 2)
print(acc1_randomforest)

X2 =  train[["Age", "Sex", "Pclass"]]
Y2 = train["Survived"]
X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2, Y2,test_size = .33, random_state=0)
randomforest = RandomForestClassifier()
randomforest.fit(X2_train, Y2_train)
Y2_pred = randomforest.predict(X2_test)
acc2_randomforest = round(accuracy_score(Y2_pred, Y2_test) * 100, 2)
print(acc2_randomforest)

X3 =  train[["Age", "Sex", "Embarked"]]
Y3 = train["Survived"]
X3_train, X3_test, Y3_train, Y3_test = train_test_split(X3, Y3,test_size = .33, random_state=0)
randomforest = RandomForestClassifier()
randomforest.fit(X3_train, Y3_train)
Y3_pred = randomforest.predict(X3_test)
acc3_randomforest = round(accuracy_score(Y3_pred, Y3_test) * 100, 2)
print(acc3_randomforest)

X4 =  train[["Age", "Sex", "Fare", "Embarked", "Pclass", "SibSp", "Parch"]]
Y4 = train["Survived"]
X4_train, X4_test, Y4_train, Y4_test = train_test_split(X4, Y4,test_size = .33, random_state=0)
randomforest = RandomForestClassifier()
randomforest.fit(X4_train, Y4_train)
Y4_pred = randomforest.predict(X4_test)
acc4_randomforest = round(accuracy_score(Y4_pred, Y4_test) * 100, 2)
print(acc4_randomforest)

score_compare = pd.DataFrame()
score_compare['Label'] =  ["[Age][Sex]", "[Age][Sex][Pclass]","[Age][Sex][Embarked]", "All Features"]
score_compare['Score'] = [acc1_randomforest, acc2_randomforest, acc3_randomforest, acc4_randomforest]

fig, ax = plt.subplots()
sns.barplot(x='Label', y = 'Score', data = score_compare, color = 'm')
plt.title('Random Forest Accuracy Score \n')
plt.ylabel('Accuracy Score')
plt.xlabel('Class Label')
ax.bar_label(ax.containers[0])


plt.show()