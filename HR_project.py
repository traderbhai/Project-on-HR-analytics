
''' This project is based on using various ensemble learning techniques
    on a given dataset of HR analytics'''

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
import xgboost as xgb
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


df=pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

#preprocessing
''' Columns under consideration'''

v=['Age','BusinessTravel', 'DailyRate', 'Department',
       'DistanceFromHome', 'Education', 'EducationField',
        'EnvironmentSatisfaction', 'Gender', 'HourlyRate',
       'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction',
       'MaritalStatus', 'MonthlyIncome', 'MonthlyRate',
       'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike',
       'PerformanceRating', 'RelationshipSatisfaction', 
       'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
       'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
       'YearsSinceLastPromotion', 'YearsWithCurrManager']
s_t_num={'No':0, 'Yes':1 }
df['Attrition']=df['Attrition'].map(s_t_num)


df.select_dtypes(['object'])

#converting categorical values to numerical values
ind_bus=pd.get_dummies(df['BusinessTravel'],prefix='BusinessTravel')
ind_dep=pd.get_dummies(df['Department'],prefix='Department')
ind_ef=pd.get_dummies(df['EducationField'],prefix='EducationField')
ind_gender=pd.get_dummies(df['Gender'],prefix='Gender')
ind_jr=pd.get_dummies(df['JobRole'],prefix='JobRole')
ind_ms=pd.get_dummies(df['MaritalStatus'],prefix='MaritalStatus')
ind_ot=pd.get_dummies(df['OverTime'],prefix='OverTime')


df1=pd.concat([ind_bus,ind_dep,ind_ef,ind_gender,ind_jr,ind_ms,ind_ot,df.select_dtypes(['int64'])],axis=1)

Y=df['Attrition']
# Division of training and test data
X_train,X_test,Y_train,Y_test= train_test_split(df1,Y,train_size=0.7,random_state=42)
def print_score(clf,X_train,Y_train,X_test, Y_test,train=True):
    #Printing the metrics of performance
    if train:
        print("\n\t\tTraining Result:\n")
        print("\naccuracy_score: \t {0:.4f}".format(accuracy_score(Y_train,clf.predict(X_train))))
        print("\nClassification_report: \n{}\n".format(classification_report(Y_train,clf.predict(X_train))))
        print("Confusion Matrix: \n {}\n".format(confusion_matrix(Y_train,clf.predict(X_train))))
        
        res=cross_val_score(clf,X_train,Y_train,cv=10,scoring="accuracy")
        print("Average Accuracy:\t {0:.4f}".format(np.mean(res)))
        print ("Accuracy SD:\t\t {0:.4f}".format(np.std(res)))
    
    elif train==False:
        print("\n\t\tTest Results:")
        print("\naccuracy_score: \t {0:.4f}".format(accuracy_score(Y_test,clf.predict(X_test))))
        print("\nClassification_report: \n{}\n".format(classification_report(Y_test,clf.predict(X_test))))
        print("\nConfusion Matrix:\n{}\n".format(confusion_matrix(Y_test,clf.predict(X_test))))

#Various models 
def Decision_Tree():
    print ("\n\t\t--- Decision Tree Classifier---\n")       
    #Scores for training data
    dc=DecisionTreeClassifier().fit(X_train,Y_train)
    print_score(dc,X_train,Y_train,X_test, Y_test,train=True)
    #Scores for test data
    print_score(dc,X_train,Y_train,X_test, Y_test,train=False)

def Bagging():
    print ("\n\t\t--- Bagging--- \n")
    #OOB=False Out of Bagging
    dc=DecisionTreeClassifier().fit(X_train,Y_train)
    bag_clf=BaggingClassifier(base_estimator=dc,n_estimators=1000,bootstrap=True,n_jobs=-1,random_state=42)
    bag_clf.fit(X_train,Y_train)
    print_score(bag_clf,X_train,Y_train,X_test, Y_test,train=True)
    #Scores for test data
    print_score(bag_clf,X_train,Y_train,X_test, Y_test,train=False)

def Random_Forest():
    print ("\n\t\t ---Random Forest Classifier----\n")       
    #Scores for training data
    rf=RandomForestClassifier().fit(X_train,Y_train)
    print_score(rf,X_train,Y_train,X_test, Y_test,train=True)
    #Scores for test data
    print_score(rf,X_train,Y_train,X_test, Y_test,train=False)
    
def extra_trees():
    print ("\n\t\t ---Extremely randomised trees Classifier---\n")       
    #Scores for training data
    et=ExtraTreesClassifier().fit(X_train,Y_train)
    print_score(et,X_train,Y_train,X_test, Y_test,train=True)
    #Scores for test data
    print_score(et,X_train,Y_train,X_test, Y_test,train=False)
def Ada_boost():
    print ("\n\t\t ---Ada Boost Classifier with random forest as base---\n")  
    rf=RandomForestClassifier()    
    #Scores for training data
    ab=AdaBoostClassifier(base_estimator=rf).fit(X_train,Y_train)
    print_score(ab,X_train,Y_train,X_test, Y_test,train=True)
    #Scores for test data
    print_score(ab,X_train,Y_train,X_test, Y_test,train=False)
def gbm():
    print ("\n\t\t ---gradient Boost Classifier with random forest as base---\n")  
    gbm=GradientBoostingClassifier()    
    #Scores for training data
    gbm.fit(X_train,Y_train)
    print_score(gbm,X_train,Y_train,X_test, Y_test,train=True)
    #Scores for test data
    print_score(gbm,X_train,Y_train,X_test, Y_test,train=False)

def extreme_gbm():
    print ("\n\t\t ---Extreme Gradient Boost Classifier---\n")  
    ex_gb=xgb.XGBClassifier(max_depth=3,n_estimators=5000,learning_rate=0.2)   
    #Scores for training data
    ex_gb.fit(X_train,Y_train)
    print_score(ex_gb,X_train,Y_train,X_test, Y_test,train=True)
    #Scores for test data
    print_score(ex_gb,X_train,Y_train,X_test, Y_test,train=False)



def main():
    Random_Forest()
    #Call the other functions just like this.
