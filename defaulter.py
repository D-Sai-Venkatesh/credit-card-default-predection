import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import warnings
from scipy.stats import norm  
import re
import matplotlib.pyplot as plt
from stop_words import get_stop_words
import nltk
from nltk.stem import PorterStemmer , WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC

warnings.filterwarnings("ignore")

data  = pd.read_csv("data.csv")

data.head()

# replacing nosence values 

data.EDUCATION.replace(to_replace =[0,6],value =5,inplace=True) 

data.MARRIAGE.replace(to_replace=0,value=3,inplace=True)

data.rename(columns={"PAY_0":"PAY_1"},inplace =True)

data.rename(columns={"default.payment.next.month":"default"},inplace =True)

data[["PAY_1","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]].replace(to_replace = -1,value =0,inplace=True) 

data[["PAY_1","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]].replace(to_replace = -2,value = 0,inplace=True) 

# plots

# data["default"].hist()

# usampled for data analysis
from sklearn.utils import resample

not_default = data[data['default']==0]
default = data[data['default']==1]

default_upsampled = resample(default,
                          replace=True, 
                          n_samples=int(len(not_default)), 
                          random_state=33) 
upsampled_analy = pd.concat([not_default, default_upsampled])
upsampled_analy = shuffle(upsampled_analy)

# fig, ax = plt.subplots()
# sns.countplot(x='SEX', hue = 'default', data=upsampled_analy, palette='Reds')

temp = upsampled_analy[upsampled_analy["SEX"]==1]
frac = temp.default.value_counts()/temp.shape[0]
frac

temp = upsampled_analy[upsampled_analy["SEX"]==2]
frac = temp.default.value_counts()/temp.shape[0]
frac

# fig, ax = plt.subplots()
# sns.countplot(x='EDUCATION', hue = 'default', data=upsampled_analy, palette='Reds')

temp = upsampled_analy[upsampled_analy["EDUCATION"]==1]
frac = temp.default.value_counts()/temp.shape[0]
frac

temp = upsampled_analy[upsampled_analy["EDUCATION"]==2]
frac = temp.default.value_counts()/temp.shape[0]
frac

temp = upsampled_analy[upsampled_analy["EDUCATION"]==3]
frac = temp.default.value_counts()/temp.shape[0]
frac
# temp.count()

temp = upsampled_analy[upsampled_analy["EDUCATION"]==4]
frac = temp.default.value_counts()/temp.shape[0]
frac
# temp.count()

temp = upsampled_analy[upsampled_analy["EDUCATION"]==5]
frac = temp.default.value_counts()/temp.shape[0]
frac
# temp.count()

# fig, axz = plt.subplots(figsize=(20,15))

# axz = sns.countplot(x='AGE', hue='default', data=upsampled_analy, palette='Reds')


# axz.set_ylabel('COUNTS', rotation=0, labelpad=40,size=20)
# axz.set_xlabel('AGE', size=20)
# axz.yaxis.set_label_coords(-0.05, 0.95)  # (x, y)
# axz.legend(loc=0,fontsize=20);

# axz.tick_params(labelsize=15) 

# fig, axz = plt.subplots(figsize=(20,15))

# axz = sns.countplot(x='LIMIT_BAL', hue='default', data=upsampled_analy, palette='Reds')


# axz.set_ylabel('COUNTS', rotation=0, labelpad=40,size=20)
# axz.set_xlabel('LIMIT_BAL', size=20)
# axz.yaxis.set_label_coords(-0.05, 0.95)  # (x, y)
# axz.legend(loc=0,fontsize=20);

# axz.tick_params(labelsize=15) 

# age is a good feature 

# fig, ax = plt.subplots()
# sns.countplot(x='MARRIAGE', hue = 'default', data=upsampled_analy, palette='Reds')

temp = upsampled_analy[upsampled_analy["MARRIAGE"]==1]
frac = temp.default.value_counts()/temp.shape[0]
frac

temp = upsampled_analy[upsampled_analy["MARRIAGE"]==2]
frac = temp.default.value_counts()/temp.shape[0]
frac

temp = upsampled_analy[upsampled_analy["MARRIAGE"]==3]
frac = temp.default.value_counts()/temp.shape[0]
frac

data.drop("ID",axis=1,inplace=True)
data.drop("MARRIAGE",axis=1,inplace=True)
data.drop("SEX",axis=1,inplace=True)
# data.drop("EDUCATION",axis=1,inplace=True)
data.drop("AGE",axis=1,inplace=True)

# data.corr()

new_temp = data.iloc[:,8:]  
new_temp.corr()

new_temp_1 = new_temp[new_temp["BILL_AMT2"]>0]
new_temp_2 = new_temp[new_temp["BILL_AMT5"]>0]
new_temp_3 = new_temp[new_temp["BILL_AMT1"]>0]
new_temp_4 = new_temp[new_temp["BILL_AMT4"]>0]

# plt.scatter(new_temp_3["BILL_AMT1"],new_temp_3["BILL_AMT2"])
# plt.show()

# #  correlation b/w bill amounts

# plt.scatter(new_temp_2["BILL_AMT2"],new_temp_2["BILL_AMT3"])
# plt.show()

# plt.scatter(new_temp_1["BILL_AMT6"],new_temp_1["BILL_AMT5"])
# plt.show()

# plt.scatter(new_temp_3["BILL_AMT1"],new_temp_3["BILL_AMT6"])
# plt.show()

# plt.scatter(new_temp_4["BILL_AMT4"],new_temp_4["BILL_AMT6"])
# plt.show()

data.drop("BILL_AMT3",axis=1,inplace=True)
data.drop("BILL_AMT5",axis=1,inplace=True)
data.drop("BILL_AMT2",axis=1,inplace=True)
# corelations are also similar



#  training model

X = data.iloc[:,:-1]  
y = data.iloc[:,-1:]
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size = 0.25, random_state=36)  
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size = 0.3, random_state=99) 

temp_X_train_val = X_train_val
temp_X_train_val["default"] = y_train_val["default"]

from sklearn.utils import resample

not_default = temp_X_train_val[temp_X_train_val['default']==0]
default = temp_X_train_val[temp_X_train_val['default']==1]

default_upsampled = resample(default,
                          replace=True, 
                          n_samples=int(len(not_default)), 
                          random_state=33) 
upsampled_data = pd.concat([not_default, default_upsampled])
upsampled_data = shuffle(upsampled_data)

X_train_upsampled_val = upsampled_data.iloc[:,:-1]  
y_train_upsampled_val = upsampled_data.iloc[:,-1:]

X_train_upsampled_val.head()
# X_train_upsampled_val.shape

# random Forest

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(class_weight=None,min_impurity_split=None,n_estimators=100,warm_start=False) 
rf.fit(X_train_upsampled_val, y_train_upsampled_val) 


predictions_rf = rf.predict(X_test)

from sklearn import metrics

# print(metrics.classification_report(y_test,predictions_rf))

repo={"accuracy":metrics.accuracy_score(y_test,predictions_rf),"precision":metrics.precision_score(y_test,predictions_rf),"recall":metrics.recall_score(y_test,predictions_rf),"f1_score":metrics.f1_score(y_test,predictions_rf)}
# print(repo)

feature_importance = pd.DataFrame(rf.feature_importances_,
                                   index = X_train.columns,
                                   columns=['Variable_Importance']).sort_values('Variable_Importance',ascending=True)
# Set seaborn contexts 
# sns.set(style="whitegrid")

# feature_importance.plot.barh(figsize=(15,10))

# feature_importance.head(30)

# precision recall trade off

def adjusted_classes(y_scores, t):
    return [1 if y >= t else 0 for y in y_scores]

predictions_rf_proba = rf.predict_proba(X_test)

x1 = np.linspace(0.2,0.9,140, endpoint = False) 

accuracy = []
precision =[]
recall =[]
f1 =[]
for x in x1:
    predictions_rf=adjusted_classes(predictions_rf_proba[:,1],x)
    accuracy.append(metrics.accuracy_score(y_test,predictions_rf))
    precision.append(metrics.precision_score(y_test,predictions_rf))
    recall.append(metrics.recall_score(y_test,predictions_rf))
    f1.append(metrics.f1_score(y_test,predictions_rf))

# plt.figure(figsize=(8, 8))
# plt.title("Precision and Recall Scores as a function of the decision threshold")
# plt.plot(x1, precision, "b--", label="Precision")
# plt.plot(x1, recall, "g-", label="Recall")
# plt.plot(x1,accuracy,"r-",label="accuracy")
# plt.plot(x1,f1,"y-",label="f1")
# plt.ylabel("Score")
# plt.xlabel("Decision Threshold")
# plt.legend(loc='best')

predictions_rf=adjusted_classes(predictions_rf_proba[:,1],0.33)
repo={"accuracy":metrics.accuracy_score(y_test,predictions_rf),"precision":metrics.precision_score(y_test,predictions_rf),"recall":metrics.recall_score(y_test,predictions_rf),"f1_score":metrics.f1_score(y_test,predictions_rf)}

print("Final Result")
print("******************************")
print(repo)
print("******************************")
print(metrics.classification_report(y_test,predictions_rf))



