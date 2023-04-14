import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

#import dataset
df=pd.read_csv("https://raw.githubusercontent.com/ryanchung403/dataset/main/train_data_titanic.csv")

#observing dataset
df.head(10)
df.info()

df.drop(['Name','Ticket'], axis=1,inplace=True)  #刪掉這兩個欄位
sns.pairplot(df[['Survived','Fare']],dropna=True)  #以下三行找關聯性
sns.pairplot(df[['Survived','Pclass']],dropna=True)
sns.pairplot(df[['Survived','Age']],dropna=True)

df.groupby('Survived').mean  ##算出生存與否各類的平均值

df['SibSp'].value_counts()   #次數
df['Parch'].value_counts()
df['Sex'].value_counts()
df['PassengerId'].value_counts()

df.isnull().sum()
len(df)/2
df.isnull().sum()>(len(df)/2)
df.drop('Cabin',axis=1,inplace=True)   #資料太少所以刪掉這欄

df['Age'].isnull().value_counts() #計算空值

df.groupby('Sex')['Age'].median().plot(kind='bar')  #用圖表示

df['Age']= df.groupby('Sex')['Age'].apply(lambda x: x.fillna(x.median()))  #空值用平均值填入，就不會影響數據。

df['Embarked'].value_counts().idxmax()  #三種種類s登船的最多
df['Embarked'].fillna(df['Embarked'].value_counts().idxmax(),inplace=True)    #處理Embarked的空值，把s種類丟進去

df=pd.get_dummies(data=df, columns=['Sex','Embarked'])
#Sex轉換數字，Embarked三種地點轉換成S,c,q
df.drop('Sex_female',axis=1,inplace=True)   

df.info()
df.corr()  ##相關性

X=df.drop(['Survived','Pclass'],axis=1)  ##把pclass比較有關聯的值丟掉
y=df['Survived']


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.3, random_state=67)  #百分之七十


#choose model & train
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(max_iter=200)  ##做分類
lr.fit(X_train,y_train)  #餵百分之70的資料
predictions= lr.predict(X_test)
#使用模型開始預測
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score
accuracy_score(y_test,predictions)
recall_score(y_test,predictions)
precision_score(y_test,predictions)
#用圖來看更清楚！
confusion_matrix(y_test,predictions)
pd.DataFrame(confusion_matrix(y_test,predictions),columns=['Predict not survived','Predict  survived'],index=['True not Survived','True Survived'])