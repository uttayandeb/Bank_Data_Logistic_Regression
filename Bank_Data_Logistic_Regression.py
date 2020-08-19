##################################################################################
############################# LOGISTIC ~ REGRESSION ##############################
##################################################################################



#### Importing packages and loading dataset ############
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from scipy import stats
import scipy.stats as st

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split # train and test 
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import classification_report

bank_data = pd.read_csv("C:\\Users\\home\\Desktop\\Data Science Assignments\\Python_codes\\Logistic_Regression\\bank-full.csv", sep=";")
#Since the data is in text format we used "sep= ;" to convert text into column

#since the  data have  O/P-“ y” is discrete in two categories and with multiple i/p variables
#so we will perform( MULTIPLE LOGISTIC REGRESSION)


###########   DATA CLEANING ########################

bank_data.head(5)
bank_data.isnull()#so there are no null or missing values in the dataset
bank_data.duplicated(subset=None, keep='first')#there are no duplicate values


############### Exploratory data analysis(EDA) ################

bank_data.describe()


# Getting the barplot for the categorical columns 

sb.countplot(x="job",data=bank_data,palette="hls")
pd.crosstab(bank_data.job,bank_data.marital).plot(kind="bar")

sb.countplot(x="education",data=bank_data,palette="hls")
pd.crosstab(bank_data.marital,bank_data.balance).plot(kind="bar")

sb.countplot(x="marital",data=bank_data,palette="hls")



bank_data.age.value_counts()
bank_data.job.value_counts()
#blue-collar      9732
#management       9458
#technician       7597
#admin.           5171
#services         4154
#retired          2264
#self-employed    1579
#entrepreneur     1487
#unemployed       1303
#housemaid        1240
#student           938
#unknown           288
#Name: job, dtype: int64


bank_data.marital.value_counts()
#married     27214
#single      12790
#divorced     5207
#Name: marital, dtype: int64

bank_data.education.value_counts()
#secondary    23202
#tertiary     13301
#primary       6851
#unknown       1857
#Name: education, dtype: int64

bank_data.default.value_counts()
#no     44396
#yes      815
#Name: default, dtype: int64

bank_data.housing.value_counts()
#yes    25130
#no     20081
#Name: housing, dtype: int64

bank_data.loan.value_counts()
#no     37967
#yes     7244
#Name: loan, dtype: int64

bank_data.contact.value_counts()
#cellular     29285
#unknown      13020
#telephone     2906
#Name: contact, dtype: int64

bank_data.month.value_counts()
#may    13766
#jul     6895
#aug     6247
#jun     5341
#nov     3970
#apr     2932
#feb     2649
#jan     1403
#oct      738
#sep      579
#mar      477
#dec      214

bank_data.poutcome.value_counts()
#unknown    36959
#failure     4901
#other       1840
#success     1511

bank_data.pdays.value_counts()
bank_data.y.value_counts()#Subscribed or not(output)
#no     39922
#yes     5289

bank1 = pd.get_dummies(bank_data, drop_first = True)
bank1.columns

plt.boxplot(bank1["age"])
age= np.cbrt(bank1["age"])
help(np.cbrt)
plt.boxplot(np.cbrt(bank1["age"]))
plt.boxplot(np.log(bank1["age"]))
ag_log =pd.DataFrame(np.log(bank1["age"]))## log of age 
ag_log.rename(columns={"age":"age_log"},inplace=True)

bank1 = pd.concat([ag_log,bank1],axis=1)
#concating log_age column with the dataset
bank1.drop(["age"],inplace = True, axis=1)
#since age column is insignificant and not needed 





x= bank1.iloc[:,0:42]#creating a new object with only taking the features or i/p variables

#x.drop(["y"],inplace= True , axis=1)


model1= sm.logit("y_yes~x", data = bank1 ).fit()
model1.summary()
model1.summary2()
## AIC:  21644.8803



#Removing all the insignificant columns which are not needed

## Majorly insignificant colmns is being removed and seeing if the insignificance of the variables are removed 

x.iloc[:,23].name
x0= x.drop(["default_yes"],axis=1)

model2= sm.logit("y_yes~x0", data = bank1).fit()
model2.summary()
model2.summary2()
 ## AIC: 21642
 
## 5 from x0 is removed as it is insignificant
x0.iloc[:,5].name
x1 = x0.drop(["pdays"],axis=1) 
#pdays: number of days that passed by after the client was last contacted from a previous campaign 

model3 = sm.logit("y_yes~x1", data =  bank1).fit()
model3.summary()
model3.summary2()
##AIC  21640

## 18 from x1 is removed as it is insignificant
x1.iloc[:,18].name
x2 = x1.drop(["marital_single"],axis=1)
model4 = sm.logit("y_yes~x2", data = bank1).fit()
model4.summary()
model4.summary2()
##AIC  21639

## 38 from x2 is removed as it is insignificant
#poutcome: outcome of the previous marketing campaign 
x2.iloc[:,38].name
x3 = x2.drop(["poutcome_unknown"],axis=1)
model5 = sm.logit("y_yes~x3", data = bank1).fit()
model5.summary()
model5.summary2()
## AIC 21638
### poutcome_unknown removed with the same logic of dummy variables. Out of 4 dummy variables, one can be removed.
## As, the model understands with 3 dummy variables.

## 16 from x3 is removed as it is insignificant
x3.iloc[:,16].name
x4 = x3.drop(["job_unknown"],axis=1)
model6 = sm.logit("y_yes~x4", data= bank1).fit()
model6.summary()
model6.summary2()
##AIC 21638

x4.iloc[:,15].name

x4.iloc[:,26].name


x5 = x4.drop(["month_feb"],axis=1)
model7 = sm.logit("y_yes~x5", data= bank1).fit()
model7.summary()
model7.summary2()
##AIC  21639

## 5 in x5 is removed, as it is insignificant 
x5.iloc[:,5].name





x6= x5.drop(["previous"], axis=1)
model8 = sm.logit("y_yes~x6", data = bank1).fit()
model8.summary()
model8.summary2()
## AIC 21641.5157

## So, considering the model8, we proceed

y_yes = pd.DataFrame(bank1["y_yes"])

y_yes.rename(columns={"y_yes":"y_yes1"},inplace= True)

x6 = pd.concat([y_yes,x6],axis=1)



st.chisqprob = lambda chisq, df:stats.chi2.sf(chisq,df)

y_pred = model8.predict(x6)
#y_pred1 = model8.predict(bank1)

x6["pred_prob"]= y_pred
x6["y_val"]= np.zeros(45211)#creating a new column and assigning it with all zeros
x6.loc[y_pred>=0.5,"y_val"]=1

#if prob value is greater then alpha value 0.5 y_val is 1 ie subscribed



from sklearn.metrics import classification_report
classification_rep = classification_report(x6["y_val"],x6["y_yes1"])


################ ACCURACY MEASURE #############
## confusion matrix
confusion_matrix= pd.crosstab(x6["y_yes1"],x6["y_val"])

##accuracy
accuracy = (38944+1829)/45211 
accuracy# 0.901838048262591
### so the model is 90.1% accurate

## ROC curve
from sklearn import metrics
##fpr=> false postive rate
##tpr=> true positive rate 

fpr,tpr,threshold = metrics.roc_curve(x6["y_yes1"],y_pred)

## plotting roc curve
plt.plot(fpr,tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")

## Area under the curve

roc_auc = metrics.auc(fpr,tpr) ##0.9076540812534906
#90.76%

### trail 2 
y_pred2= model8.predict(x6)

###
x6["pred_prob2"]=y_pred2
x6["y_val2"]= np.zeros(45211)

x6.loc[y_pred2>=0.82,"y_val2"]=1

## Classification report
classification_rep2 = classification_report(x6["y_val2"],x6["y_yes1"])

##confusion matrix
confusion_matrix2 = pd.crosstab(x6["y_yes1"],x6["y_val2"])

##accuracy
accuracy2= (39651+664)/45211##  0.8917077702329079,89%
#so this time we are getting less accuracy 
## The cutoff point is 0.5, above which the accuracy decreases,



x6.drop(["pred_prob","y_val","pred_prob2","y_val2"],inplace= True, axis=1)
x6 = x6.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,0]]


###############################################################################
########### SPLITTING THE DATA INTO TRAIN AND TEST DATA #######################
###############################################################################

x6.rename(columns={"y_yes1":"y_yes"},inplace=True)
from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(x6)


### As we are getting an error of mismatch of rows between X6 and train data, we modify the model as,

x_train = train_data.iloc[:,1:36]
##training data
model81 = sm.logit("y_yes~x_train", data = train_data).fit()
model81.summary()

## prediction on training data
y_train = model81.predict(train_data)

train_data["pred_train"]=y_train
train_data["train_vals"]=np.zeros(33908)
train_data.loc[y_train>=0.5,"train_vals"]=1

##confusion matrix
confusion_matrix_train = pd.crosstab(train_data["y_yes"],train_data["train_vals"])

## accuracy

accuracy = (29929+3979)/ (29929+3979)  #1.0


##prediction on test data
x_test = test_data.iloc[:,1:36]

model82 = sm.logit("y_yes~x_test", data =test_data).fit()
model82.summary()

## prediction
y_test = model82.predict(test_data)
test_data["pred_test"]=y_test
test_data["test_vals"] = np.zeros(11303)
test_data.loc[y_test>=0.5,"test_vals"]=1

## confusion matrix

confusion_matrix_test = pd.crosstab(test_data["y_yes"],test_data["test_vals"])

## Accuracy
accuracy_test= (9954+865)/(9954+54+430+865) # 0.9571795098646377
##95.72%