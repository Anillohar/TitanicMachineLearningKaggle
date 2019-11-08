#!/usr/bin/env python
# coding: utf-8

# <h1><center>TITANIC survivor classification</center></h1>

# <h5><b>Importing the libraries</b></h5>

# In[76]:


# linear algebra
import numpy as np 
# data processing
import pandas as pd 

# data visualization
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
from matplotlib import style

# Algorithms
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB


# <h5><b>Getting the Data</b></h5>

# In[77]:


test_df = pd.read_csv("../data/test.csv")
train_df = pd.read_csv("../data/train.csv")


# In[78]:


train_df.info()


# In[79]:


train_df.describe()


# In[80]:


train_df.head(8)


# <h5>Imputing Missing values</h5>

# In[81]:


# Total number of null values with columns
total = train_df.isnull().sum().sort_values(ascending=False)
total


# In[82]:


percent_1 = train_df.isnull().sum()/train_df.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
percent_2


# In[83]:


missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
missing_data


# In[84]:


train_df.columns.values


# <h5>Visualization Part</h5>

# In[85]:


survived = 'survived'
not_survived = 'not survived'

fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))

women = train_df[train_df['Sex']=='female']
men = train_df[train_df['Sex']=='male']

ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde =False)
ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde =False)
ax.legend()
ax.set_title('Female')

ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde = False)
ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde = False)
ax.legend()
_ = ax.set_title('Male')


# In[86]:


FacetGrid = sns.FacetGrid(train_df, row='Embarked', height=4.5, aspect=1.6)
FacetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette=None,  order=None, hue_order=None )
FacetGrid.add_legend()


# In[87]:


sns.barplot(x='Pclass', y='Survived', data=train_df)
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', height=2.2, aspect=1.6)
grid.add_legend()


# In[88]:


data = [train_df, test_df]
for dataset in data:
    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']
    dataset.loc[dataset['relatives'] > 0, 'not_alone'] = 0
    dataset.loc[dataset['relatives'] == 0, 'not_alone'] = 1
    dataset['not_alone'] = dataset['not_alone'].astype(int)
train_df['not_alone'].value_counts()


# In[89]:


axes = sns.catplot('relatives','Survived', data=train_df, aspect = 2.5, kind='point')


# <h5>Data preprocessing</h5>

# In[15]:


#missing data
#drop passenger id not important
train_df = train_df.drop(['PassengerId'], axis=1)

import re
deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}

data = [train_df, test_df]

for dataset in data:
    dataset['Cabin'] = dataset['Cabin'].fillna("U0")
    dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
    dataset['Deck'] = dataset['Deck'].map(deck)
    dataset['Deck'] = dataset['Deck'].fillna(0)
    dataset['Deck'] = dataset['Deck'].astype(int)

dataset


# In[16]:


# we can now drop the cabin feature
train_df = train_df.drop(['Cabin'], axis=1)
test_df = test_df.drop(['Cabin'], axis=1)


# <b><p>age</b></p>

# In[17]:


data = [train_df, test_df]

# the iteration will take place twice 1st for train_df, 2nd for test_df

for dataset in data:
    mean = train_df["Age"].mean()            #mean of the ages
    std = test_df["Age"].std()               #standard deviation of ages
    is_null = dataset["Age"].isnull().sum()  # checking how many columns are null

    # compute random numbers between the mean, std and is_null
    rand_age = np.random.randint(mean - std, mean + std, size = is_null)
    
    # fill NaN values in Age column with random values generated
    age_slice = dataset["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    dataset["Age"] = age_slice
    
    # defining type of all age columns to be integers
    dataset["Age"] = train_df["Age"].astype(int)      
    
train_df["Age"].isnull().sum()


# <b><p>Embarked</b></p>

# In[18]:


train_df['Embarked'].describe()


# In[19]:


common_value = 'S'
data = [train_df, test_df]

for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].fillna(common_value)
    


# <b><p>converting features</b></p>

# In[20]:


train_df.info()


# <b><p>Fare</b></p>

# In[21]:


data = [train_df, test_df]

for dataset in data:
    dataset['Fare'] = dataset['Fare'].fillna(0)
    dataset['Fare'] = dataset['Fare'].astype(int)


# <b><p>Name</b></p>

# In[22]:


data = [train_df, test_df]
titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in data:
    # extract titles
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    
    # replace titles with a more common title or as Rare
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
    # convert titles into numbers
    dataset['Title'] = dataset['Title'].map(titles)
    
    # filling NaN with 0, to get safe
    dataset['Title'] = dataset['Title'].fillna(0)
    
train_df = train_df.drop(['Name'], axis=1)
test_df = test_df.drop(['Name'], axis=1)


# <b><p>Sex</b></p>

# In[23]:


genders = {"male": 0, "female": 1}
data = [train_df, test_df]

for dataset in data:
    dataset['Sex'] = dataset['Sex'].map(genders)
    


# <b><p>Ticket</b></p>

# In[24]:


train_df['Ticket'].describe()


# In[25]:


train_df = train_df.drop(['Ticket'], axis=1)
test_df = test_df.drop(['Ticket'], axis=1)


# <b><p>Embarked (converting into unique)</b></p>

# In[26]:


ports = {"S": 0, "C": 1, "Q": 2}
data = [train_df, test_df]

for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].map(ports)


# <h5>Creating Categories</h5>

# In[27]:


#AGE
data = [train_df, test_df]
for dataset in data:
    dataset['Age'] = dataset['Age'].astype(int)
    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5
    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6
    dataset.loc[ dataset['Age'] > 66, 'Age'] = 6

# let's see how it's distributed train_df['Age'].value_counts()


# In[28]:


#FARE
train_df.head(10)

data = [train_df, test_df]

for dataset in data:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare']   = 3
    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare']   = 4
    dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5
    dataset['Fare'] = dataset['Fare'].astype(int)
    


# <h5>Creating New Features</h5>

# In[29]:


#AGE TIMES CLASS
data = [train_df, test_df]
for dataset in data:
    dataset['Age_Class']= dataset['Age']* dataset['Pclass']
    


# In[30]:


#FARE PER PERSON
for dataset in data:
    dataset['Fare_Per_Person'] = dataset['Fare']/(dataset['relatives']+1)
    dataset['Fare_Per_Person'] = dataset['Fare_Per_Person'].astype(int)
    
# Let's take a last look at the training set, before we start training the models.
train_df.head(10)


# <h5>Building ML models</h5>

# In[31]:


# dropping survived column from train DF and Passenger id from test DF
X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()


# <p><b> 1. Stochastic Gradient Descent (SGD) </b></p>

# In[87]:


stochastic_gradient_descent = linear_model.SGDClassifier(max_iter=5, tol=None)
stochastic_gradient_descent.fit(X_train, Y_train)

Y_pred = stochastic_gradient_descent.predict(X_test)
accuracy_sgd = round(stochastic_gradient_descent.score(X_train, Y_train) * 100, 2)
accuracy_sgd


# <p><b> 2. Random Forest Classification </b></p>

# In[88]:


random_forest = RandomForestClassifier(n_estimators=100)    # 100 number of trees
random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)
accuracy_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
accuracy_random_forest


# <p><b>3. Logistic Regression</b></p>

# In[89]:


logistic_regression = LogisticRegression(solver='warn')
logistic_regression.fit(X_train, Y_train)

Y_pred = logistic_regression.predict(X_test)
accuracy_logistic_regression = round(logistic_regression.score(X_train, Y_train) * 100, 2)
accuracy_logistic_regression


# <p><b>4. K Nearest Neighbor

# In[90]:


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)
accuracy_knn = round(knn.score(X_train, Y_train) * 100, 2)
accuracy_knn


# <p><b>5. Gaussian Naive Bayes

# In[91]:


gaussian_naive_bayes = GaussianNB() 
gaussian_naive_bayes.fit(X_train, Y_train)
Y_pred = gaussian_naive_bayes.predict(X_test)

accuracy_gaussian_naive_bayes = round(gaussian_naive_bayes.score(X_train, Y_train) * 100, 2)
accuracy_gaussian_naive_bayes


# <p><b>6. Decision Tree Classification

# In[92]:


decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)
accuracy_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
accuracy_decision_tree


# <p><b>7. Linear Support Vector Machine

# In[100]:


linear_svc = LinearSVC(max_iter=10000)
linear_svc.fit(X_train, Y_train)

Y_pred = linear_svc.predict(X_test)

accuracy_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
accuracy_linear_svc


# <p><b>8. Perceptron

# In[101]:


perceptron = Perceptron(max_iter=100)
perceptron.fit(X_train, Y_train)

Y_pred = perceptron.predict(X_test)

accuracy_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
accuracy_perceptron


# <h5>Finding the best model

# In[103]:


results = pd.DataFrame({
    'Model': [
              'Stochastic Gradient Decent', 'Random Forest Classification', 'Logistic Regression',
              'KNN', 'Gaussian Naive Bayes', 'Decision Tree Classification', 'Linear Support Vector Machines',  
              'Perceptron', 
             ],
    'Score': [
              accuracy_sgd, accuracy_random_forest, accuracy_logistic_regression, accuracy_knn,
              accuracy_gaussian_naive_bayes, accuracy_decision_tree, accuracy_linear_svc, accuracy_perceptron
             ]
                    })
result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df


# <h5>K Fold Cross validation

# In[106]:


from sklearn.model_selection import cross_val_score

rf = RandomForestClassifier(n_estimators=100)
scores = cross_val_score(rf, X_train, Y_train, cv=10, scoring = "accuracy")

print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())


# <h5>Finding importance of all the features

# In[109]:


importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(random_forest.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances


# In[110]:


importances.plot.bar()


# <h5>removing the features with p value less than 0.05

# In[111]:


train_df  = train_df.drop("not_alone", axis=1)
test_df  = test_df.drop("not_alone", axis=1)

train_df  = train_df.drop("Parch", axis=1)
test_df  = test_df.drop("Parch", axis=1)


# In[113]:


random_forest = RandomForestClassifier(n_estimators=100, oob_score = True)
random_forest.fit(X_train, Y_train)
Y_prediction = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

accuracy_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
round(accuracy_random_forest,2,)


# In[114]:


print("oob score:", round(random_forest.oob_score_, 4)*100, "%")


# <h5>Lets nown tune the hyperparameters for random forest</h5>
# <p>criterion, min_samples_leaf, min_samples_split and n_estimators

# In[ ]:


param_grid = { "criterion" : ["gini", "entropy"], 
               "min_samples_leaf" : [1, 5, 10, 25, 50, 70], 
               "min_samples_split" : [2, 4, 10, 12, 16, 18, 25, 35], 
               "n_estimators": [100, 400, 700, 1000, 1500]
             }

from sklearn.model_selection import GridSearchCV
rf = RandomForestClassifier(n_estimators=100, max_features='auto', oob_score=True, random_state=1, n_jobs=-1)

clf = GridSearchCV(estimator=rf, param_grid=param_grid, n_jobs=-1, cv=3)
clf.fit(X_train, Y_train)
clf.bestparams


# In[117]:


clf.cv_results_


# In[119]:


clf.best_params_


# In[32]:


# Using best parameters
random_forest = RandomForestClassifier(criterion = "entropy", 
                                       min_samples_leaf = 1, 
                                       min_samples_split = 16,   
                                       n_estimators=700, 
                                       max_features='auto', 
                                       oob_score=True, 
                                       random_state=1, 
                                       n_jobs=-1)

random_forest.fit(X_train, Y_train)
Y_prediction = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

print("oob score:", round(random_forest.oob_score_, 4)*100, "%")


# <h5>Model Evaluation

# <p><b>1. Confusion Matrix

# In[38]:


from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

predictions = cross_val_predict(random_forest, X_train, Y_train, cv=3, n_jobs=4)
confusion_matrix(Y_train, predictions)


# <p><b>2. Precision and Recall

# In[43]:


from sklearn.metrics import precision_score, recall_score

# Precision tells us that model predicts 83% of the time, a passengers survival correctly
print("Precision:", round(precision_score(Y_train, predictions)*100, 4), '%')

# Recall tells us that it predicted the survival of 73 % of the people who actually survived
print("Recall:",round(recall_score(Y_train, predictions)*100, 4), '%')


# <p><b>3. F1 Score</p></b>
# <p>The F1-score is computed with the harmonic mean of precision and recall.it assigns much more weight to low values. As a result of that, the classifier will only get a high F1-score, if both recall and precision are high.

# In[46]:


from sklearn.metrics import f1_score
print('F1-score', round(f1_score(Y_train, predictions)*100, 4), '%')


# <p><pre>F1-score is 77% which is not good for us
# but because it <strong>favors classifiers that have a similar precision and recall</strong>. 
# This is a problem, because you sometimes want a high precision and sometimes a high recall. 
# The thing is that an increasing precision, sometimes results in an decreasing recall and vice versa (depending on the threshold). This is called the <strong>precision/recall tradeoff.</strong></pre></p>

# <h5>Precision Recall Tradeoff

# <p><b>1. Precision Recall Curve

# In[56]:


from sklearn.metrics import precision_recall_curve

# getting the probabilities of our predictions
y_scores = random_forest.predict_proba(X_train)
y_scores = y_scores[:,1]
precision, recall, threshold = precision_recall_curve(Y_train, y_scores)


# In[61]:


def plot_precision_and_recall(precision, recall, threshold):
    plt.plot(threshold, precision[:-1], "r-", label="precision", linewidth=5)
    plt.plot(threshold, recall[:-1], "b", label="recall", linewidth=5)
    plt.xlabel("threshold", fontsize=19)
    plt.legend(loc="upper right", fontsize=19)
    # ylim to adjust the graph notations to show everything
    plt.ylim([0, 1])

plt.figure(figsize=(17, 7))
plot_precision_and_recall(precision, recall, threshold)
plt.show()


# In[64]:


def plot_precision_vs_recall(precision, recall):
    plt.plot(recall, precision, "g", linewidth=2.5)
    plt.ylabel("recall", fontsize=19)
    plt.xlabel("precision", fontsize=19)
    plt.axis([0, 1.5, 0, 1.5])

plt.figure(figsize=(17, 7))
plot_precision_vs_recall(precision, recall)
plt.show()


# <p><b>2. ROC AUC Curve</b></p>
# <strong>True Positives v/s False Positives</strong>
# <pre>This curve plots the true positive rate (also called recall) against 
# the false positive rate (ratio of incorrectly classified negative instances), 
# instead of plotting the precision versus the recall.</pre>

# In[70]:


from sklearn.metrics import roc_curve

# compute true positive rate and false positive rate
false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_train, y_scores)

# plotting them against each other
def plot_roc_curve(false_positive_rate, true_positive_rate, label=None):
    plt.plot(false_positive_rate, true_positive_rate, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'r', linewidth=2)
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate (FPR)', fontsize=16)
    plt.ylabel('True Positive Rate (TPR)', fontsize=16)

plt.figure(figsize=(14, 7))
plot_roc_curve(false_positive_rate, true_positive_rate)
plt.show()


# <pre>The red line in the middel represents a purely random classifier (e.g a coin flip) and therefore your classifier should be as far away from it as possible.
# we also have a tradeoff here, because the classifier produces more false positives, the higher the true
# positive rate is.
# </pre>

# <p><b>3. ROC AUC Score</p></b>
# <pre>The ROC AUC Score is the corresponding score to the ROC AUC Curve. It is simply computed by measuring the
# area under the curve, which is called AUC.
# A classifiers that is 100% correct, would have a ROC AUC Score of 1 and a completely random classiffier 
# would have a score of 0.5.</pre>

# In[75]:


from sklearn.metrics import roc_auc_score
r_a_score = roc_auc_score(Y_train, y_scores)
print("ROC-AUC-Score:", round(r_a_score*100, 4))

