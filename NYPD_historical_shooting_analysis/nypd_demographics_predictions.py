#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 19:59:20 2024

@author: allisonforsyth
"""

import os
import numpy as np
os.getcwd()
os.chdir('/Users/allisonforsyth/Documents/GMU_Spring/CS_504')

import pandas as pd
df = pd.read_csv("nypd.csv")
df.head()

df.PERP_RACE.unique()

df = df[df['PERP_RACE'] != "UNKNOWN"]
df = df[df['PERP_RACE'] != "(null)"]
df = df[df['PERP_SEX'] != "U"]
df = df[df['PERP_AGE_GROUP'] != "UNKNOWN"]
df = df[df['VIC_AGE_GROUP'] != "UNKNOWN"]
df = df[df['VIC_RACE'] != "UNKNOWN"]
df = df[df['VIC_SEX'] != "U"]

df.dropna(subset=['PERP_RACE', "PERP_AGE_GROUP", "PERP_SEX"], inplace=True)
df

df["STATISTICAL_MURDER_FLAG"] = df["STATISTICAL_MURDER_FLAG"].astype(int)

x1 = df["STATISTICAL_MURDER_FLAG"]
x2 = df["Latitude"]
x3 = df["Longitude"]

df = df[["PERP_RACE", "PERP_AGE_GROUP", "PERP_SEX", "VIC_RACE", "VIC_AGE_GROUP", "VIC_SEX" , "BORO","STATISTICAL_MURDER_FLAG", "Latitude", "Longitude"]]

df_feat = df[["VIC_RACE", "VIC_AGE_GROUP", "VIC_SEX" , "BORO"]]
df_feat = pd.get_dummies(df_feat, dtype=int)
df_feat= df_feat.join(x1)
df_feat= df_feat.join(x2)
df_feat= df_feat.join(x3)
df_feat.to_numpy()

list(df_feat.columns)


X = df_feat
y1 = df["PERP_RACE"].to_numpy()
y2 = df["PERP_SEX"].to_numpy()
y3 = df["PERP_AGE_GROUP"].to_numpy()
Y = np.vstack((y1, y2, y3)).T
Y

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
     X, Y, test_size=0.30, random_state=42)


from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

#Warning: At present, no metric in sklearn.metrics supports the multioutput-multiclass classification task.


n_samples, n_features = X.shape # 10,100
n_outputs = Y.shape[1] # 3
n_classes = 3
forest = RandomForestClassifier(random_state=1)
multi_target_forest = MultiOutputClassifier(forest, n_jobs=2)
multi_target_forest.fit(X_train, y_train)

pred = multi_target_forest.predict(X_test)
y_test

pred = pd.DataFrame(pred, columns =['racepred', 'sexpred', 'agepred'])
test = pd.DataFrame(y_test, columns =['race1', 'sex1', 'age1'])
pred
test
pred_df = pd.concat([pred, test], axis=1)
pred_df
pred_df['race_true'] = (pred_df['racepred']==pred_df['race1']).astype(int)
pred_df['sex_true'] = (pred_df['sexpred']==pred_df['sex1']).astype(int)
pred_df['age_true'] = (pred_df['agepred']==pred_df['age1']).astype(int)


pred_df['true_all']=pred_df.race_true + pred_df.age_true + pred_df.sex_true

pred_df.race_true.sum()/4182
pred_df.age_true.sum()/4182
pred_df.sex_true.sum()/4182

pred_df['true_all'].value_counts()

#race sex boro age
three_true = 1616/4182
two_true = 1932/4182
one_true = 617/4182
none_true = 17/4182

three_true 
two_true
one_true 
none_true 

three_true + two_true + one_true 

#seperate predictions to get metrics
race_pred = pred_df["racepred"].to_numpy()
race_true = pred_df["race1"].to_numpy()

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

accuracy_score(race_true, race_pred)
confusion_matrix(race_true, race_pred)

matrix = confusion_matrix(race_true, race_pred)
matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

print(classification_report(race_true, race_pred))

import matplotlib.pyplot as plt
import seaborn as sns
# Build the plot
plt.figure(figsize=(16,7))
sns.set(font_scale=1.4)
sns.heatmap(matrix, annot=True, annot_kws={'size':10},
            cmap=plt.cm.Greens, linewidths=0.2)

# Add labels to the plot
class_names = ['American Indian/Alaskan Native','Asain/Pacific Islander', 'Black','Black Hispanic', 'White', 'White Hispanic']
tick_marks = np.arange(len(class_names))
tick_marks2 = tick_marks + 0.5
plt.xticks(tick_marks, class_names, rotation=25)
plt.yticks(tick_marks2, class_names, rotation=0)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix for Random Forest Model: Race')
plt.show()


#sex
sex_pred = pred_df["sexpred"].to_numpy()
sex_true = pred_df["sex1"].to_numpy()

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

accuracy_score(sex_true, sex_pred)
confusion_matrix(sex_true, sex_pred)

matrix = confusion_matrix(sex_true, sex_pred)
matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

print(classification_report(sex_true, sex_pred))

import matplotlib.pyplot as plt
import seaborn as sns
# Build the plot
plt.figure(figsize=(16,7))
sns.set(font_scale=1.4)
sns.heatmap(matrix, annot=True, annot_kws={'size':10},
            cmap=plt.cm.Greens, linewidths=0.2)

# Add labels to the plot
class_names = ['Female','Male']
tick_marks = np.arange(len(class_names))
tick_marks2 = tick_marks + 0.5
plt.xticks(tick_marks, class_names, rotation=25)
plt.yticks(tick_marks2, class_names, rotation=0)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix for Random Forest Model: Sex')
plt.show()

#seperate predictions to get metrics
age_pred = pred_df["agepred"].to_numpy()
age_true = pred_df["age1"].to_numpy()

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

accuracy_score(age_true, age_pred)
confusion_matrix(age_true, age_pred)

matrix = confusion_matrix(age_true, age_pred)
matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]


print(classification_report(age_true, age_pred))

import matplotlib.pyplot as plt
import seaborn as sns
# Build the plot
plt.figure(figsize=(16,7))
sns.set(font_scale=1.4)
sns.heatmap(matrix, annot=True, annot_kws={'size':10},
            cmap=plt.cm.Greens, linewidths=0.2)

# Add labels to the plot
class_names = ['18-24','25-44', '45-64','65+', '<18']
tick_marks = np.arange(len(class_names))
tick_marks2 = tick_marks + 0.5
plt.xticks(tick_marks, class_names, rotation=25)
plt.yticks(tick_marks2, class_names, rotation=0)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix for Random Forest Model: Age')
plt.show()





















#from sklearn.metrics import accuracy_score
#accuracy_score(y_test, pred)

#all factors race sex boro age murder
three_true = 1554/4181
two_true = 1954/4181
one_true = 655/4181
none_true = 18/4181

three_true 
two_true
one_true 
none_true 

#race sex age boro murder
pred_df.race_true.sum()/4181
pred_df.age_true.sum()/4181
pred_df.sex_true.sum()/4181

three_true = 1169/4181
two_true = 2261/4181
one_true = 728/4181
none_true = 23/4181

three_true 
two_true
one_true 
none_true 









#"corelation" age
df2 = pd.read_csv("nypd.csv")
df2.head()


df2 = df2[df2['PERP_AGE_GROUP'] != "UNKNOWN"]
df2 = df2[df2['VIC_AGE_GROUP'] != "UNKNOWN"]

df2.dropna(subset=["PERP_AGE_GROUP"], inplace=True)
df2


df2["sameage"]=df2.VIC_AGE_GROUP==df2.PERP_AGE_GROUP
df2


df2["sameage"] = df2["sameage"].astype(int)
df2.sameage.sum()

6782/14766
#0.45929838818908303

#"corelation" race
df3 = pd.read_csv("nypd.csv")


df3 = df3[df3['PERP_RACE'] != "UNKNOWN"]
df3 = df3[df3['PERP_RACE'] != "(null)"]
df3 = df3[df3['VIC_RACE'] != "UNKNOWN"]


df3.dropna(subset=["PERP_RACE"], inplace=True)
df3


df3["samerace"]=df3.VIC_RACE==df3.PERP_RACE
df3


df3["samerace"] = df3["samerace"].astype(int)
df3.samerace.sum()

10615/15484
#0.6855463704469129

df3 = df3[df3['PERP_RACE'] == 'BLACK']
df3 = df3[df3['VIC_RACE'] == 'BLACK']
df3
#black/black
9059/10615
#0.8534149788035799

df3 = df3[df3['PERP_RACE'] == 'WHITE']
df3 = df3[df3['VIC_RACE'] == 'WHITE']
df3
#white/white
157/10615
#0.014790390956194065

df3 = df3[df3['PERP_RACE'] == 'BLACK HISPANIC']
df3 = df3[df3['VIC_RACE'] == 'BLACK HISPANIC']
df3
#black hispanic
344/10615
#0.0324069712670749

df3 = df3[df3['PERP_RACE'] == 'WHITE HISPANIC']
df3 = df3[df3['VIC_RACE'] == 'WHITE HISPANIC']
df3
#white hispanic
1003/10615
#0.0944889307583608

df3 = df3[df3['PERP_RACE'] == 'ASIAN / PACIFIC ISLANDER']
df3 = df3[df3['VIC_RACE'] == 'ASIAN / PACIFIC ISLANDER']
df3
#asian
52/10615
#0.004898728214790391

df3 = df3[df3['PERP_RACE'] == 'AMERICAN INDIAN/ALASKAN NATIVE']
df3 = df3[df3['VIC_RACE'] == 'AMERICAN INDIAN/ALASKAN NATIVE']
df3
#none


#"corelation" sex
df4 = pd.read_csv("nypd.csv")


df4 = df4[df4['PERP_SEX'] != "U"]
df4 = df4[df4['VIC_SEX'] != "U"]

df4.dropna(subset=["PERP_SEX"], inplace=True)
df4


df4["samesex"]=df4.VIC_SEX==df4.PERP_SEX
df4


df4["samesex"] = df4["samesex"].astype(int)
df4.samesex.sum()

13839/16496
#0.8389306498545102

#male
df4 = df4[df4['PERP_SEX'] == "M"]
df4 = df4[df4['VIC_SEX'] == "M"]
df4
13767/13839
#0.9947973119445047

#female
df4 = df4[df4['PERP_SEX'] == "F"]
df4 = df4[df4['VIC_SEX'] == "F"]
df4

72/13839
#0.00520268805549534

#One classifier
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
     X, y1, test_size=0.30, random_state=42, stratify=y1)

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(max_depth=7, random_state=0)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

accuracy_score(y_test, pred)
confusion_matrix(y_test, pred)

matrix = confusion_matrix(y_test, pred)
matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

pred = pd.DataFrame(pred, columns =['racepred'])
test = pd.DataFrame(y_test, columns =['racetrue'])
pred
test
pred_df = pd.concat([pred, test], axis=1)

print(classification_report(y_test, pred))

import matplotlib.pyplot as plt
import seaborn as sns
# Build the plot
plt.figure(figsize=(16,7))
sns.set(font_scale=1.4)
sns.heatmap(matrix, annot=True, annot_kws={'size':10},
            cmap=plt.cm.Greens, linewidths=0.2)

# Add labels to the plot
class_names = ['18-24','25-44', '45-64','65+', '<18']
tick_marks = np.arange(len(class_names))
tick_marks2 = tick_marks + 0.5
plt.xticks(tick_marks, class_names, rotation=25)
plt.yticks(tick_marks2, class_names, rotation=0)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix for Random Forest Model')
plt.show()
















