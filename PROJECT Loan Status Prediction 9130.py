#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# PROJECT LOAN STATUS PREDICTION USING MACHINE LEARNING algorithm 


# In[1]:


# Eligibility prediction loan 


# # Data set Information
# Dream Housing Finance company deals in all home loans. they have a presence across all Urban,semi_Urban and rural areas.
# the customber first applies  for a home loan and after that ,the company validates the customer eligibility for the loan.
# the company wants to automate  the loan eligibilty process (real_time) based on customber detail provided while filling out 
# online application forms.these details are Gender,Marital status,Education,number of Dependents Income ,LoanAmount,
# Credit History,and others. to automatic this process ,they have  provided a dataset to identify the customer segments that are 
# eligible for loan amounts so that they can specifically target these customers
# 
#      this binary classification problem in which we need to predict our target label which is "loan Status 
# 
# 
#     Loan Status have two values : Yes or No
# 
# YES: if the loan is approved
# NO: if the loan is not approved
# 
# 
# Data set Description
#  There are 13 attributes/Features in this data set:
# 
# 8 categorical Features,
# 4 continuous Features
# 1 variable to accommodate the loan ID.
# 
# # The machine learning models used in this project are:
# 
# Logistic Regression
# Support Vector Machine (SVM)
# Decision Tree
# Naive Bayes 
# Random Forest
# 
# 
# 
# 

# In[136]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns


# In[137]:


loan=pd.read_csv("C:/Users/Akash/Downloads/archive (69).zip")


# In[138]:


loan


# In[139]:


loan.shape


# # Data Preparation and Cleaning

# In[140]:


# printing the first five rows are dataframe
loan.head()


# In[141]:


loan.tail()


# In[142]:


loan.info()


# # there are 614 rows and  12 columns are in the given data set

# In[143]:


# statastical measure 

loan.describe()


# In[144]:


# number of missing value in each column
loan.isnull().sum()


# In[145]:


loan.columns


# In[146]:


# label encoding
loan["Loan_Status"].replace("N",0,inplace=True)
loan["Loan_Status"].replace("Y",1,inplace=True)


# In[147]:


loan["Loan_Status"].value_counts()


# In[148]:


loan['Gender'].fillna(value=loan['Gender'].mode()[0],inplace=True)
loan['Married'].fillna(loan['Married'].mode()[0],inplace= True)
loan['Dependents'].fillna(loan['Dependents'].mode()[0],inplace= True)
loan['Self_Employed'].fillna(loan['Self_Employed'].mode()[0],inplace= True)
loan['Credit_History'].fillna(loan['Credit_History'].mode()[0],inplace=True)
loan['Loan_Amount_Term'].fillna(loan['Loan_Amount_Term'].mode()[0],inplace=True)
loan['LoanAmount'].fillna(loan['LoanAmount'].median(),inplace=True)


# In[149]:


# cleaning the missing value in each column
loan.isnull().sum()


# In[150]:


loan.head()


# In[151]:


# dependent column value
loan["Dependents"].value_counts()


# In[152]:


# replacing the value of 3+ to 4
loan=loan.replace(to_replace="3+",value=3)


# In[153]:


loan["Dependents"].value_counts()


# # Data visualization
# 

# In[154]:


# heatmap


plt.figure(figsize=(12,5))
sns.heatmap(loan.corr(), annot=True, cmap='Greys')

  target variable(Loan_Status) has strong Positive correlation with the Applicantincome and Loan amount variable
 Also, we can note a feeble positive correlation between the credit history and few other  self_Employed ,
Gender,Married,Educ

# In[155]:


# Education & Loan status
sns.countplot(x="Education",hue="Loan_Status",data=loan)


# In[156]:


#  The number of loans approved is more for graduated people than people having no graduation.


# In[157]:


# Married & Loan Status
sns.countplot(x="Married",hue="Loan_Status",data=loan)


# In[158]:


#  The number of loans approved is more for married people than people who are not married.


# In[159]:


# convert categerical columns to Numerical values
loan.replace({"Married":{"No":0,"Yes":1},"Gender":{"Male":1,"Female":0},"Self_Employed":{"No":0,"Yes":1},
             "Property_Area":{"Rural":0,"Semiurban":1,"Urban":2},"Education":{"Graduate":1,"Not Graduate":0}},inplace=True)


# In[160]:


loan.head()


# In[161]:


loan.boxplot(rot=90,figsize=(25,10))


# In[162]:


loan.boxplot(column='ApplicantIncome')


# In[163]:


loan=loan[loan.ApplicantIncome<7900]


# In[164]:


loan.ApplicantIncome.plot(kind="box")


# In[165]:


loan['ApplicantIncome'].hist()


# In[166]:


loan['LoanAmount'].hist()


# In[167]:


loan.boxplot(column='ApplicantIncome',by ='Education')


# In[168]:


loan.boxplot(column='LoanAmount')


# In[169]:


loan=loan[loan.LoanAmount<200]


# In[170]:


loan.LoanAmount.plot(kind="box")


# In[171]:


loan=loan[loan.LoanAmount>30]


# In[172]:


loan.LoanAmount.plot(kind="box")


# In[173]:


loan=loan[loan.CoapplicantIncome>9.8]


# In[174]:


loan.CoapplicantIncome.plot(kind="box",figsize=(20,10))


# In[175]:


# numerical attributes visualization
sns.distplot(loan["ApplicantIncome"])


# In[176]:


# apply log transformation to the attribute
loan["ApplicantIncome"]=np.log(loan["ApplicantIncome"])


# In[177]:


sns.distplot(loan["ApplicantIncome"])


# In[178]:


sns.distplot(loan["CoapplicantIncome"])


# In[179]:


loan["CoapplicantIncome"]=np.log(loan["CoapplicantIncome"])


# In[180]:


sns.distplot(loan["CoapplicantIncome"])


# In[183]:


loan["LoanAmount"]=np.log(loan["LoanAmount"])


# In[186]:


# separting the data and labe
x=loan.drop(columns=["Loan_ID","Loan_Status"],axis=1)
y=loan["Loan_Status"]


# In[187]:


x


# In[188]:


y


# In[189]:


# Spliting the data for train,test


# In[190]:


from sklearn.model_selection import train_test_split


# In[191]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[192]:


x_train.shape,x_test.shape


# In[193]:


y_train.shape,y_test.shape


# # LogisticRegression

# In[194]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


# In[195]:


model=LogisticRegression()


# In[196]:


model.fit(x_train,y_train)


# In[197]:


pred=model.predict(x_test)


# In[198]:


pred


# In[202]:


lr=accuracy_score(y_test,pred)


# In[203]:


lr


# In[204]:


confusion_matrix(y_test,pred)


# In[205]:


(10+41)/(10+6+1+41)


# In[206]:


print(classification_report(y_test,pred))


# # from LogisticRegrssion model the accuracy_score is 87.93%

# # support vector machine
# 

# In[216]:


from sklearn import svm


# In[219]:


svm.SVC(kernel='linear')


# In[220]:


classifier.fit(x_train, y_train)


# In[221]:


# accuracy_score of training data set
train_prediction = classifier.predict(x_train)
print(f"The accuracy for test data is : {accuracy_score(train_prediction, y_train)}")


# In[222]:


test_prediction = classifier.predict(x_test)
print(f"The accuracy for test data is : {accuracy_score(test_prediction, y_test)}")


# In[227]:


confusion_matrix(test_prediction,y_test)


# In[228]:


(10+41)/(10+1+6+41)


#  # from Support vector machine the accuracy score is 87.93%

# # naive Bayes algorithm

# In[229]:


from sklearn.naive_bayes import GaussianNB
NBClassifier=GaussianNB()
NBClassifier.fit(x_train,y_train)


# In[230]:


y_pred=NBClassifier.predict(x_test)


# In[231]:


y_pred


# In[232]:


accuracy_score(y_pred,y_test)


# In[233]:


confusion_matrix(y_pred,y_test)


# In[235]:


(11+36)/(11+6+5+36)


# # from naive bayes algorithm the accuracy score is 81.03% 

# # Decision Tree Classifier

# In[236]:


from sklearn.tree import DecisionTreeClassifier


# In[237]:


model=DecisionTreeClassifier()


# In[238]:


model.fit (x_train,y_train)


# In[239]:


pred=model.predict(x_test)


# In[240]:


pred


# In[245]:


accuracy_score(y_test,pred)


# In[242]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(20,15))
from sklearn import tree
tr = tree.plot_tree(model,filled=True,fontsize=10)


# In[246]:


confusion_matrix(y_test,pred)


# In[247]:


(10+34)/(10+6+8+34)


# # from Decision Tree Classifier the accuracy score is 75.86%

# # random forest classifier 
# 

# In[248]:


from sklearn.ensemble import RandomForestClassifier


# In[249]:


model1=RandomForestClassifier()


# In[250]:


model1.fit(x_train,y_train)


# In[251]:


pred=model1.predict(x_test)


# In[252]:


pred


# In[253]:


accuracy_score(y_test,pred)


# In[254]:


confusion_matrix(y_test,pred)


# In[258]:


(10+40)/(10+6+2+40)


# In[255]:


print(classification_report(y_test,pred))


# # from Random Forest  Classifier the accuracy score is 86.20%¶
#model comparision
 model                          accuracy_score
 Logistic Regression               87.93%
 SVM                              87.93 %
 Navie bayes                       81.03%
 Decision tree                     75.86%                   
Random Forest                     86.20%

from above result we can seen that the Logistic Regression,SVM,Navie bayes,RandomForest Can achive 80% Accuracy score 
and Decision Tree have a less Accuracy Score is the 75% 
The highest accuracy is 87.93% (LogisticRgression And support Vector Machine)
