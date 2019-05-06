# TItanic-survival-prediction
# Building a Logistic regression model using the data containing various features after needed data cleaning and exploring for correlated features.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

train = pd.read_csv(r'C:\Users\Avante Team1\Downloads\Data for ML\titanic_train.csv')


# Exploratory Data Analysis:
# I did some exploratory data analysis and started by checking out missing data.

# Missing Data:
# I used seaborn to create a simple heatmap to see where is the missing data.

sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

# Roughly 20 percent of the Age data is missing. The proportion of Age missing is likely small enough for reasonable replacement with some form of imputation. Looking at the Cabin column, it looks like we are just missing too much of that data to do something useful with at a basic level.

# Some more VIsualising is needed

sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train,palette='RdBu_r')


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')

sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')



sns.distplot(train['Age'].dropna(),kde=False,color='darkred',bins=30)



train['Age'].hist(bins=30,color='darkred',alpha=0.7)



sns.countplot(x='SibSp',data=train)



train['Fare'].hist(color='green',bins=40,figsize=(8,4))


# Data Cleaning
# I needed to fill in missing age data instead of just dropping the missing age data rows. One way to do this was by filling in the mean age of all the passengers.

plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')

# We could see from the boxplot that the wealthier passengers in the higher classes tend to be older, which makes sense. I used these average age values to impute based on Pclass for Age.


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age
       
# and now applied this function

train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)

# Now heat map, again!

sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

# Dropping the Cabin column and the row in Embarked that is NaN.

train.drop('Cabin',axis=1,inplace=True)

train.dropna(inplace=True)


# Converting Categorical Features:
# We needed to convert categorical features to dummy variables using pandas.

sex = pd.get_dummies(train['Sex']) #,drop_first=True)

embark = pd.get_dummies(train['Embarked'],drop_first=True)
train = pd.concat([train,sex,embark],axis=1)


train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)

# Now the data was ready for our model.


# Building a Logistic Regression model
# Started by splitting our data into a training set and test set

from sklearn.model_selection import train_test_split

X = train.drop('Survived',axis=1)
y = train['Survived']

X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1),train['Survived'], test_size=0.30,random_state=101)


# Training and Predicting

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()

predictions = logmodel.predict(X_test)


# Evaluation:
# The last thing to do was to check precision, recall and f1-score using classification report!

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,predictions))

print(confusion_matrix(y_test,predictions))

logmodel.fit(X_train,y_train)
