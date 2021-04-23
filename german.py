import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
by=pd.read_csv(r'/Users/bhaskaryuvaraj/Library/Mobile Documents/com~apple~CloudDocs/Documents/data_science work_files/german_data.csv',encoding='latin')

by.columns
by.head()
by.dtypes
by['Status of existing checking account']=by['Status of existing checking account'].str.replace(r'\D','').astype(int)
by['Credit history'].unique()

#---------------------------------------------EDA-------------------------------------------------------------
by.groupby(['Status of existing checking account','Default_status'])['Default_status'].count().plot(kind='bar')
#most people with checking account status of A11 has taken the loan

by.groupby(['Credit history','Default_status'])['Default_status'].count().plot(kind='bar')
#most people with credit history of A32 has adopted

by.groupby(['Purpose','Default_status'])['Default_status'].count().plot(kind='bar')
#most people with A40 purpose has adopted

by['Credit amount'].hist(by=by['Default_status'])
#from the graph it is clear that people with credit amount btw 0-5000 have mostly opted

by.groupby(['Property','Default_status'])['Default_status'].count().plot(kind='bar')
#proper with A123 have mostly accepted


by['Age in Years'].unique()
by.groupby([pd.cut(x=by['Age in Years'],bins=(19,29,39,49,59,69,79,89)),'Default_status'])['Default_status'].count().plot(kind='bar')
#mostly people btw age of 19-29 have opted

by.groupby(['Job_status','Default_status'])['Default_status'].count().plot(kind='bar')
#most people with job status of A173 have opted

by['Housing'].unique()
by.columns=by.columns.str.strip()
by.groupby(['Housing','Default_status'])['Default_status'].count().plot(kind='bar')
#most people with housing of A152 have opted in

by.groupby(['Number of existing credits at this bank','Default_status'])['Default_status'].count().plot(kind='bar')
#most people with 1 no. of existinfg credit have opted in

by.groupby(['Job_status','Default_status'])['Default_status'].count().plot(kind='bar')
#most people with job status of A173 have optedin

by['foreign worker'].unique()
by.groupby(['foreign worker','Default_status'])['Default_status'].count().plot(kind='bar')
#mostly foreign worker of A201 have opted in

#---------------------------------------------------EDA ends-------------------------------------------------

#to check for missing values
by.isnull().sum()
#no null values

#to check for outliers
by.boxplot('Duration in month')
by.boxplot('Credit amount')
by.boxplot('Age in Years')


#to remove the outliers
def remove_outlier(d,c):
    q1=d[c].quantile(0.25)
    q3=d[c].quantile(0.75)
    iqr=q3-q1
    ub=q3+1.53*iqr
    lb=q1-1.53*iqr
    result=d[(d[c]>lb) & (d[c]<ub)]
    return result

by=remove_outlier(by,'Duration in month')
by.boxplot('Duration in month')
by=remove_outlier(by,'Credit amount')
by.boxplot('Credit amount')
by=remove_outlier(by,'Age in Years')
by.boxplot('Age in Years')

#now removing A from the columns and converting it into int from object so that nothing will be affected
by['Status of existing checking account']=by['Status of existing checking account'].str.replace(r'\D','').astype(int)
by['Credit history']=by['Credit history'].str.replace(r'\D','').astype(int)
by['Purpose']=by['Purpose'].str.replace(r'\D','').astype(int)
by['Savings account/bonds']=by['Savings account/bonds'].str.replace(r'\D','').astype(int)
by['Present employment since']=by['Present employment since'].str.replace(r'\D','').astype(int)
by['Personal status and sex']=by['Personal status and sex'].str.replace(r'\D','').astype(int)
by['Other debtors / guarantors']=by['Other debtors / guarantors'].str.replace(r'\D','').astype(int)
by['Property']=by['Property'].str.replace(r'\D','').astype(int)
by['Other installment plans']=by['Other installment plans'].str.replace(r'\D','').astype(int)
by['Housing']=by['Housing'].str.replace(r'\D','').astype(int)
by['Job_status']=by['Job_status'].str.replace(r'\D','').astype(int)
by['Telephone']=by['Telephone'].str.replace(r'\D','').astype(int)
by['foreign worker']=by['foreign worker'].str.replace(r'\D','').astype(int)

#to find the correlation
correlated_features = set()
correlation_matrix = by.drop('Default_status', axis=1).corr()

for i in range(len(correlation_matrix.columns)):

    for j in range(i):

        if abs(correlation_matrix.iloc[i, j]) > 0.8:

            colname = correlation_matrix.columns[i]

            correlated_features.add(colname)
#Check correlated features            
print(correlated_features)
#none of the features are corelated

#seperating dependent and independent variables
x=by.drop('Default_status',axis=1)
y=by['Default_status'].copy()

#create training and test data by splitting x and y into 70:30 ratio
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=8)

#logistic regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train, y_train)

print(logreg.score(x_train,y_train))
#accuracy=0.7889908256880734
pred_y=logreg.predict(x_test)

#accuracy of test model
logreg.score(x_test,y_test)
# accuracy=0.7777777777777778





