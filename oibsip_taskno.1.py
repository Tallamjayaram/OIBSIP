import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report, confusion_matrix,accuracy_score

from sklearn.tree import plot_tree
data = pd.read_csv('Iris.csv')
print(data)

print(data.isnull().any())

print(data.shape)

sns.pairplot(data=data, hue = 'Species')

sns.heatmap(data.corr())

target =data['Species']
data1 = data.copy()
data1 = data1.drop('Species', axis =1)
print(data1.shape)

#label encoding
le = LabelEncoder()
target = le.fit_transform(target)
print(target)

X = data1
y = target

X_train, X_test, y_train, y_test = train_test_split(X , y, test_size = 0.25, random_state = 42)

print("Training split input- ", X_train.shape)
print("Testing split input- ", X_test.shape)


dtree=DecisionTreeClassifier()
dtree.fit(X_train,y_train)

print('Decision Tree Classifer Created')

y_pred = dtree.predict(X_test)
print("Accuracy of the model is:",accuracy_score(y_pred,y_test)*100)
print("Classification report - \n", classification_report(y_test,y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,5))
sns.heatmap(data=cm,linewidths=.5, annot=True,square = True,  cmap = 'Blues')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(dtree.score(X_test, y_test))
plt.title(all_sample_title, size = 15)


plt.figure(figsize = (20,20))
dec_tree = plot_tree(decision_tree=dtree, feature_names = data1.columns, 
                     class_names =["setosa", "vercicolor", "verginica"] , filled = True , precision = 4, rounded = True)
