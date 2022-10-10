
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score , mean_squared_error,confusion_matrix , classification_report
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
import streamlit as st
from PIL import Image
import pickle
import warnings
warnings.filterwarnings('ignore')



st.write("""
# **Bootcamp Final Project**:
## Credit card payment default prediction
#### *The aim of this project will be to predict the the payment default for the next months for the given credits based on different demographic variables. In order to achieve it, we are going to analyze the influence of each variable and we are going to use different machine learning algorithms to get the best possible predictions.*
""")


image=Image.open('D:\WB\Ironhack\Final_Project\credit\credit_card.png')

st.image(image)

st.write("""
#### INFORMATION ABOUT THE DATA

*LIMIT_BAL*: Amount of the given credit (NT dollar): it includes both the individual consumer credit and his/her family (supplementary) credit.

*SEX*: Gender (1 = male; 2 = female).

*EDUCATION*: Education (1 = graduate school(Master or Doctor degree's); 2 = university; 3 = high school; 4 = others).

*MARRIAGE*: Marital status (1 = married; 2 = single; 3 = others).

*AGE*: Age (year).

*PAY_1-PAY_6*: History of past payment. We tracked the past monthly payment recordsfrom April to September, 2005 as follows:

PAY_1 = the repayment status in September, 2005; PAY_2 = the repayment status in August, 2005; . . .;PAY_6 = the repayment status in April, 2005. The measurement scale for the repayment status is:

-2: No consumption; -1: Paid in full; 0: The use of revolving credit; 1 = payment delay for one month; 2 = payment delay for two months; . . .; 8 = payment delay for eight months; 9 = payment delay for nine months and above.

*BILL_AMT1–BILL_AMT6*: Amount of bill statement (NT dollar).

BILL_AMT1 = amount of bill statement in September 2005, BILL_AMT2 = amount of bill statement in August 2005, BILL_AMT6 = amount of bill statement in April 2005.

*PAY_AMT1-PAY_AMT6*: Amount of previous payment (NT dollar).

PAY_AMT1 = amount paid in September 2005, PAY_AMT2 = amount paid in August 2005, PAY_AMT6 = amount paid in April 2005.

**dpnm**: default payment next month. Our target variable (our *y*). The value we want to predict. y=0 then not default, y=1 then default.

""")


pd.set_option('display.max_columns', None)


df_credit=pd.read_csv("default of credit card clients.csv")


categorical_feat=['sex', 'education','marriage'] 
numeric_feat=['limit_bal','age','repay_sept','repay_aug','bill_amt_sept','pay_amt_sept','pay_amt_aug','pay_amt_july','pay_amt_june','pay_amt_may','pay_amt_april']

st.sidebar.write("""
### Please, introduce the following features:
""")
# Categorical variables

sex_var=st.sidebar.selectbox('Is the client male or female?',['male','female'])
edu_var=st.sidebar.selectbox('What is the education of the client?',['university','graduate_school','high_school','unknown','other'])
marr_var=st.sidebar.selectbox('Which is the marital status of the client?',['married','single','other','unknown'])

cat_var_list=[sex_var,edu_var,marr_var]

# Numeric variables

limit_bal_var = st.sidebar.number_input('Introduce the amount of the given credit',min_value=10000, max_value=1000000)
age_var = st.sidebar.number_input('Introduce the age of the client',min_value=18, max_value=80, step=1) 
repay_sept_var = st.sidebar.number_input('Introduce the repay time for September',min_value=-2, max_value=8) 
repay_aug_var =st.sidebar.number_input('Introduce the repay time for August',min_value=-2, max_value=8)
bill_amt_sept_var =st.sidebar.number_input('Introduce the amount that the client has to pay in September',min_value=-165580, max_value=964511)
pay_amt_sept_var =st.sidebar.number_input('Introduce the amount that client paid in September',min_value=0, max_value=896040)
pay_amt_aug_var =st.sidebar.number_input('Introduce the amount that client paid in August',min_value=0, max_value=1684259)
pay_amt_july_var =st.sidebar.number_input('Introduce the amount that client paid in July',min_value=0, max_value=896040)
pay_amt_june_var =st.sidebar.number_input('Introduce the amount that client paid in June',min_value=0, max_value=896040)
pay_amt_may_var =st.sidebar.number_input('Introduce the amount that client paid in May',min_value=0, max_value=896040)
pay_amt_april_var =st.sidebar.number_input('Introduce the amount that client paid in April',min_value=0, max_value=896040)

num_var_list=[limit_bal_var,age_var,repay_sept_var,repay_aug_var,bill_amt_sept_var,pay_amt_sept_var,pay_amt_aug_var,pay_amt_july_var,pay_amt_june_var,pay_amt_may_var,pay_amt_april_var]

# Categorical and numerica dataframes

data_cat={'sex':[sex_var],'education':[edu_var],'marriage':[marr_var]}
data_num={'limit_bal':[limit_bal_var],'age':[age_var],'repay_sept':[repay_sept_var],'repay_aug':[repay_aug_var],'bill_amt_sept':[bill_amt_sept_var],'pay_amt_sept':[pay_amt_sept_var],'pay_amt_aug':[pay_amt_aug_var],'pay_amt_july':[pay_amt_july_var],'pay_amt_june':[pay_amt_june_var],'pay_amt_may':[pay_amt_may_var],'pay_amt_april':[pay_amt_april_var]}




loaded_encoder = pickle.load(open('encoder_model.sav', 'rb'))
loaded_scaled = pickle.load(open('scaled_model.sav', 'rb'))




# Dataframes with the values given by the user

df_cat_user=pd.DataFrame(data_cat)
df_num_user=pd.DataFrame(data_num)

# Scaling numeric df given by the user

# scaler_user=StandardScaler()
df_num_user_sc=loaded_scaled.transform(df_num_user)

# Encoding categorical df given by the user

df_cat_user_enc = loaded_encoder.transform(df_cat_user).toarray()

# Creating the X_test to predict the result with the data provided by the user
#df_num_user_sc
#df_cat_user_enc
X_test_pred=np.concatenate([df_num_user_sc,df_cat_user_enc],axis=1)


loaded_model = pickle.load(open('finalized_model.sav', 'rb'))


y_tl_pred_dt = loaded_model.predict(X_test_pred) 

path=loaded_model.decision_path(X_test_pred)
#path

probab=loaded_model.predict_proba(X_test_pred)
probab
# Hacer el "st.write" con el valor "y_tl_pred_dt" para mostrar el valor de la predicción realizada.

if y_tl_pred_dt=='yes':
        st.write("""
                        #### The client will default the payment next month
                        """)

elif y_tl_pred_dt=='no':
         st.write("""
                        #### The client will not default the payment next month
                        """)