import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
plt.style.use('seaborn')

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = load_iris()

X = data.data 
y = data.target


# random forest

clf = RandomForestClassifier()
clf.fit(X, y)

#print(accuracy_score(y, clf.predict(X)))

importance = clf.feature_importances_ 

idx = importance.argsort()[::-1]

importance2 = importance[idx]
label = [data.feature_names[i] for i in idx]

plt.bar(label, importance2, width=0.5, color='r')
plt.xticks(label, rotation=90)
#plt.show()



# Extra important features and re-run RF

from sklearn.feature_selection import SelectFromModel 

selector = SelectFromModel(clf, threshold=0.3)
x_imp = selector.fit_transform(X, y)

clf_imp = RandomForestClassifier().fit(x_imp, y)

importance_3 = clf_imp.feature_importances_ 

print(importance_3)





