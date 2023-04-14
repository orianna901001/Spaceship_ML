import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


#import dataset
df=pd.read_csv("https://raw.githubusercontent.com/ryanchung403/dataset/main/Housing_Dataset_Sample.csv")

#observing dataset
df.head(10)
df.describe().T
sns.distplot(df['Price'])
sns.jointplot(x=df['Avg. Area Income'],y=df['Price'])
sns.pairplot(df)

#prepare to train model
X =df.iloc[:,:5]
Y=df['Price']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test =train_test_split(X,Y,test_size=0.3,random_state=54)

#choose model & train
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,y_train)

#use model
predictions=reg.predict(X_test)
predictions
y_test

#evaluate model
from sklearn.metrics import r2_score
r2_score(y_test,predictions)

plt.scatter(y_test,predictions,color='blue',alpha=0.1)



