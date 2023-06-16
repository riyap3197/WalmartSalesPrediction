import pandas as pd
import datetime                                        # To handle dates
#import calendar                                        # To get month
# import statsmodels.formula.api as sm
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
# import sklearn.metrics as metrics                      # To get regression metrics
# import scipy as sp
# import time                                            # To do time complexity analysis
# import random
import copy
# import profile
# import cProfile
# from sklearn.cluster import KMeans                     # perform clustering operation
from datetime import datetime
# from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import AdaBoostRegressor
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.svm import SVC, LinearSVC

# =============================================================================
# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)
# =============================================================================

# Data preprocessing:
#loading in raw data
features_df = pd.read_csv("features.csv")
stores_df = pd.read_csv("stores.csv")
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print(features_df.head())
print(stores_df.head())
print(train_df)
print(test_df)



# merging the data
 
# =============================================================================
#  (train + Store + Feature)
#  (test + Stoee + Feature)
#  
# =============================================================================

# =============================================================================
# train_bt = pd.merge(train_df,stores_df) 
# train_df = pd.merge(train_bt,features_df)
# 
# test_bt = pd.merge(test_df,stores_df)
# test_df= pd.merge(test_bt,features_df)
# 
# =============================================================================
# =============================================================================
# print(features_df.head())
# print(features_df.describe())
# 
# print(train_df.head())
# print(train_df.describe())
# print(train_df.tail())
# 
# =============================================================================
# =============================================================================
# print(test_df.head(2))
# print(test_df.describe())
# 
# print(train_df.info())
# =============================================================================



# Creating a custom season dictionary to identify the season in each month
seasons_dict = {
    1:"Winter",
    2:"Winter",
    3:"Spring",
    4:"Spring",
    5:"Spring",
    6:"Summer", 
    7:"Summer",
    8:"Summer",
    9:"Fall",
    10:"Fall",
    11:"Fall",
    12:"Winter"
}

test_bt = pd.merge(test_df,stores_df)
test_df= pd.merge(test_bt,features_df)

# Creating the master dataset  ((train + Store + Feature))
master_df = train_df.merge(stores_df, on='Store', how='left')
master_df = master_df.merge(features_df, on=['Store', 'Date'], how='left')

d = copy.deepcopy(master_df)

d1 = d["Weekly_Sales"]

print(d["Weekly_Sales"].describe())

print("Percentile less than 3% provides only negative value : ",d["Weekly_Sales"].quantile(0.003))


x = np.concatenate((d1[d["Weekly_Sales"] < 0], d1[d["Weekly_Sales"] > 0]))

plt.hist(x, density=True)

plt.xlim([-70496, 200000])
plt.xlabel('Weekly Sales Values')
plt.ylabel('Normalized Sales Values')
plt.title('Normalized distribution of sales values')
plt.show()

print(master_df.head())

# Filling empty markdown columns
master_df['MarkDown1'] = master_df['MarkDown1'].fillna(0)
master_df['MarkDown2'] = master_df['MarkDown2'].fillna(0)
master_df['MarkDown3'] = master_df['MarkDown3'].fillna(0)
master_df['MarkDown4'] = master_df['MarkDown4'].fillna(0)
master_df['MarkDown5'] = master_df['MarkDown5'].fillna(0)

# =============================================================================
# # Cleaning holiday columns
master_df['isHoliday'] = master_df['IsHoliday_x']
master_df = master_df.drop(columns=['IsHoliday_x', 'IsHoliday_y'])
master_df['Date'] = pd.to_datetime(master_df['Date'], format='%Y-%m-%d')
master_df['Year'] = master_df['Date'].dt.year

    #  store vs sales
ax= sns.barplot(x="Store", y="Weekly_Sales",  data=master_df)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=7)
plt.tight_layout()
plt.show()

# sales vs type : year wise
sns.barplot(x="Year", y="Weekly_Sales", hue="Type", data=master_df)
 
df_corr = master_df.corr()
ax=df_corr[['Weekly_Sales']].plot(kind='bar')
plt.xlabel('Attribute')
plt.ylabel('Correlation')
plt.title('Correlation of Weekly sales with other variables')
plt.tight_layout()
plt.show()

sns.heatmap(df_corr)


train_corr=pd.DataFrame(master_df.corr())
# train_corr.to_excel(writer,'Train_Data Corr',index=True)
print(train_corr.head())

test_corr=pd.DataFrame(test_df.corr())
# train_corr.to_excel(writer,'Train_Data Corr',index=True)
print(train_corr.head())

# line graph of store vs sales 
master_df.plot(kind='line', x='Weekly_Sales', y='Store', alpha=0.5)
plt.show()

# bar graph of store vs sales
master_df['Store'].value_counts(normalize=True).plot(kind = 'bar',fig=(4,5))



# Sales vs Deptartment

master_df.plot(kind='line', x='Dept', y='Weekly_Sales', alpha=1.5,fig=(4,5))
plt.show()

# Missing Value Treatment
print(master_df.isnull().sum())
print("*"*30)
print(test_df.isnull().sum())


test_df['CPI']=test_df.groupby(['Dept'])['CPI'].transform(lambda x: x.fillna(x.mean()))
test_df['Unemployment']=test_df.groupby(['Dept'])['Unemployment'].transform(lambda x: x.fillna(x.mean()))

test_df=test_df.fillna(0)


print(master_df.isnull().sum())
print("*"*30)
print(test_df.isnull().sum())

# Outlier Treatment
master_df.Weekly_Sales=np.where(master_df.Weekly_Sales>100000, 100000,master_df.Weekly_Sales)
master_df.Weekly_Sales.plot.hist(bins=25)
plt.show()

master_df.info()

master_df['Date'] = pd.to_datetime(master_df['Date'])
test_df['Date'] = pd.to_datetime(test_df['Date'])


# Extract date features
master_df['Date_dayofweek'] =master_df['Date'].dt.dayofweek
master_df['Date_month'] =master_df['Date'].dt.month 
master_df['Date_year'] =master_df['Date'].dt.year
master_df['Date_day'] =master_df['Date'].dt.day 
master_df['IsHoliday'] =master_df['isHoliday'] 

test_df['Date_dayofweek'] =test_df['Date'].dt.dayofweek 
test_df['Date_month'] =test_df['Date'].dt.month 
test_df['Date_year'] =test_df['Date'].dt.year
test_df['Date_day'] =test_df['Date'].dt.day


print(master_df.Type.value_counts())
print("*"*30)
print(test_df.Type.value_counts())

print(train_df.IsHoliday.value_counts())
print("*"*30)
print(test_df.IsHoliday.value_counts())

train_test_data = [master_df, test_df]

type_mapping = {"A": 1, "B": 2, "C": 3}
for dataset in train_test_data:
    dataset['Type'] = dataset['Type'].map(type_mapping)
    
    
# Converting Categorical Variable 'IsHoliday' into Numerical Variable 
    
type_mapping = {False: 0, True: 1}
for dataset in train_test_data:
    dataset['IsHoliday'] = dataset['IsHoliday'].map(type_mapping)
    
    
# Creating Extra Holiday Variable.
# If that week comes under extra holiday then 1(=Yes) else 2(=No)
    
master_df['Super_Bowl'] = np.where((master_df['Date']==datetime(2010, 2, 12)) | (master_df['Date']==datetime(2011, 2, 11)) | (master_df['Date']==datetime(2012, 2, 10)) | (master_df['Date']==datetime(2013, 2, 8)),1,0)
master_df['Labour_Day'] = np.where((master_df['Date']==datetime(2010, 9, 10)) | (master_df['Date']==datetime(2011, 9, 9)) | (master_df['Date']==datetime(2012, 9, 7)) | (master_df['Date']==datetime(2013, 9, 6)),1,0)
master_df['Thanksgiving'] = np.where((master_df['Date']==datetime(2010, 11, 26)) | (master_df['Date']==datetime(2011, 11, 25)) | (master_df['Date']==datetime(2012, 11, 23)) | (master_df['Date']==datetime(2013, 11, 29)),1,0)
master_df['Christmas'] = np.where((master_df['Date']==datetime(2010, 12, 31)) | (master_df['Date']==datetime(2011, 12, 30)) | (master_df['Date']==datetime(2012, 12, 28)) | (master_df['Date']==datetime(2013, 12, 27)),1,0)
#........................................................................
test_df['Super_Bowl'] = np.where((test_df['Date']==datetime(2010, 2, 12)) | (test_df['Date']==datetime(2011, 2, 11)) | (test_df['Date']==datetime(2012, 2, 10)) | (test_df['Date']==datetime(2013, 2, 8)),1,0)
test_df['Labour_Day'] = np.where((test_df['Date']==datetime(2010, 9, 10)) | (test_df['Date']==datetime(2011, 9, 9)) | (test_df['Date']==datetime(2012, 9, 7)) | (test_df['Date']==datetime(2013, 9, 6)),1,0)
test_df['Thanksgiving'] = np.where((test_df['Date']==datetime(2010, 11, 26)) | (test_df['Date']==datetime(2011, 11, 25)) | (test_df['Date']==datetime(2012, 11, 23)) | (test_df['Date']==datetime(2013, 11, 29)),1,0)
test_df['Christmas'] = np.where((test_df['Date']==datetime(2010, 12, 31)) | (test_df['Date']==datetime(2011, 12, 30)) | (test_df['Date']==datetime(2012, 12, 28)) | (test_df['Date']==datetime(2013, 12, 27)),1,0)

# Altering the isHoliday value depending on these new holidays...
master_df['IsHoliday']=master_df['IsHoliday']|master_df['Super_Bowl']|master_df['Labour_Day']|master_df['Thanksgiving']|master_df['Christmas']
test_df['IsHoliday']=test_df['IsHoliday']|test_df['Super_Bowl']|test_df['Labour_Day']|test_df['Thanksgiving']|test_df['Christmas']


print(master_df.Christmas.value_counts())
print(master_df.Super_Bowl.value_counts())
print(master_df.Thanksgiving.value_counts())
print(master_df.Labour_Day.value_counts())

print(test_df.Christmas.value_counts())
print(test_df.Super_Bowl.value_counts())
print(test_df.Thanksgiving.value_counts())
print(test_df.Labour_Day.value_counts())

# Since we have Imputed IsHoliday according to Extra holidays..These extra holiday variable has redundant..
# Droping the Extra holiday variables because its redundant..
dp=['Super_Bowl','Labour_Day','Thanksgiving','Christmas']
master_df.drop(dp,axis=1,inplace=True)
test_df.drop(dp,axis=1,inplace=True)

print(master_df.info())
master_df.head(2)
# Since we have imputed markdown variables therefore we will not be removing the all markdown variables.
# -Removing MarkDown5 because its Highly Skewed
pd.set_option('display.max_columns', 10)
features_drop=['Unemployment','CPI','MarkDown5','isHoliday','Year']
features_drop_test=['Unemployment','CPI','MarkDown5']
master_df=master_df.drop(features_drop, axis=1)
test_df=test_df.drop(features_drop_test, axis=1)

print(master_df.head(2))
print(test_df.head(2))

# Classification & Accuracy
# Define training and testing set

# Converting all float var int integer..
for var in master_df:
    if master_df[var].dtypes == float:
        master_df[var]=master_df[var].astype(int)
        
for var in test_df:
    if test_df[var].dtypes == float:
        test_df[var]=test_df[var].astype(int)

#### train X= Exery thing except Weekly_Sales
master_df_X=master_df.drop(['Weekly_Sales','Date'], axis=1)

#### train Y= Only Weekly_Sales 
master_df_y=master_df['Weekly_Sales'] 
test_df_X=test_df.drop('Date',axis=1).copy()

print(master_df_X.shape, master_df_y.shape, test_df_X.shape)

# Building models & comparing their RMSE values
# 1.Linear Regression
# print(master_df_X)
## Methood 1..
clf = LinearRegression()
clf.fit(master_df_X, master_df_y)
y_pred_linear=clf.predict(test_df_X)
acc_linear=round( clf.score(master_df_X, master_df_y) * 100, 2)
print ('scorbe:'+str(acc_linear) + ' percent')


# 2. Random Forest
clf = RandomForestRegressor(n_estimators=100)
clf.fit(master_df_X, master_df_y)
y_pred_rf=clf.predict(test_df_X)
acc_rf= round(clf.score(master_df_X, master_df_y) * 100, 2)
print ("Accuracy: %i %% \n"%acc_rf)

# 3. Decision tree
clf=DecisionTreeRegressor()
clf.fit(master_df_X, master_df_y)
y_pred_dt= clf.predict(test_df_X)
acc_dt = round( clf.score(master_df_X, master_df_y) * 100, 2)
print (str(acc_dt) + ' percent')

# =============================================================================
# 
# Comparing Models
# Let's compare the accuracy score of all the regression models used above.
# =============================================================================


models = pd.DataFrame({
    'Model': ['Linear Regression','Random Forest','Decision Tree'],
    
    'Score': [acc_linear, acc_rf,acc_dt]
    })

print(models.sort_values(by='Score', ascending=False))

# Prediction value using Random Forest model..
submission = pd.DataFrame({
        "Store_Dept_Date": test_df.Store.astype(str)+'_'+test_df.Dept.astype(str)+'_'+test_df.Date.astype(str),
        "Weekly_Sales": y_pred_rf
    })

# =============================================================================
# submission.to_csv('weekly_sales predicted.csv', index=False)
# submission.to_excel(pd.writer,'Weekly_sales Pred',index=False)
# 
# =============================================================================
print(submission.head())    