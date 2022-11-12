#Naive Bayes used for find diabetes\

#Importing all Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix


#import dataset
df=pd.read_csv("Diabetes_RF.csv")

x=df.iloc[:,0:8]
y=df.iloc[:,-1]

#train and test
X_train, X_test, y_train, y_test =train_test_split(x,y,test_size=0.3,random_state=0)

#Gaussian Naive Bayes
Gmodel=GaussianNB()

classifier=Gmodel.fit(X_train,y_train)

y_Pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_Pred)

classifier.score(X_test,y_Pred)
print(classifier.score(X_test,y_Pred))
