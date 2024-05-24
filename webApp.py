import streamlit as st
import pandas as pd
import numpy as np
import sklearn
import pandas as pd
import numpy as np

#import data
url = "https://opendata.arcgis.com/api/v3/datasets/680b093be7274fc8a2b92756c38499bd_0/downloads/data?format=csv&spatialRefId=4326&where=1%3D1"
df = pd.read_csv(url)

#Select relavent features
df = df[['HH_WIRED_BROADBAND', 'RESP_EDU_4_CATEGORIES', 'HH_INCOME', 'HH_ADULTS_COUNT', 'HH_18_OR_UNDER_COUNT', 'HH_COMBINED_PHONE_ACCESS', 'HH_K_12_HOUSEHOLD', 'HH_TABLET']]

#data cleaning the target feature
df = df[df['HH_WIRED_BROADBAND'] != 8]
df = df[df['HH_WIRED_BROADBAND'] != 9]
df.loc[df['HH_WIRED_BROADBAND'] == 1, 'HH_WIRED_BROADBAND'] = -1
df.loc[df['HH_WIRED_BROADBAND'] == 2, 'HH_WIRED_BROADBAND'] = 1
df.loc[df['HH_WIRED_BROADBAND'] == -1, 'HH_WIRED_BROADBAND'] = 0

df = df[df['RESP_EDU_4_CATEGORIES'] != 9]

df['INCOME_IDK'] = (df['HH_INCOME'] == 98).astype(int)

df['INCOME_REFUSED'] = (df['HH_INCOME'] == 99).astype(int)

df['HH_INCOME'].replace(98, np.nan, inplace=True)
df['HH_INCOME'].replace(99, np.nan, inplace=True)

df['HH_ADULTS_COUNT'].replace(9, np.nan, inplace=True)
df['HH_ADULTS_COUNT'].replace(8, np.nan, inplace=True)

df = df[df['HH_18_OR_UNDER_COUNT'] != 98]
df = df[df['HH_18_OR_UNDER_COUNT'] != 99]

df = df[df['HH_TABLET'] != 8]
df = df[df['HH_TABLET'] != 9]
df.loc[df['HH_TABLET'] == 1, 'HH_TABLET'] = -1
df.loc[df['HH_TABLET'] == 2, 'HH_TABLET'] = 0
df.loc[df['HH_TABLET'] == -1, 'HH_TABLET'] = 1

#train test split
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(df, test_size=0.15, random_state = 20)

#training labels
training_labels = train_set.copy()
training_labels = train_set['HH_WIRED_BROADBAND']

#training features
training_features = train_set.copy()
training_features.drop(columns=['HH_WIRED_BROADBAND'], inplace=True)

#testing lables
testing_labels = test_set.copy()
testing_labels = test_set['HH_WIRED_BROADBAND']

#testing features
testing_features = test_set.copy()
testing_features.drop(columns=['HH_WIRED_BROADBAND'], inplace=True)

from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.compose import ColumnTransformer

imputer = SimpleImputer(strategy='median')

#standardize pipeline
standardize_pipeline = make_pipeline(imputer,
                                     StandardScaler())

#OneHotEncode pipeline
onehot_pipeline = make_pipeline(imputer,
                                OneHotEncoder())

#sqrt and standardize pipeline
sqrt_standardize_pipeline = make_pipeline(imputer,
                                          FunctionTransformer(np.sqrt, feature_names_out="one-to-one"),
                                          StandardScaler())

#sum, log, and standardize pipeline
def household_total(X):
  X_np = X#.values  # Convert DataFrame to NumPy array
  return X_np[:, [0]] + X_np[:, [1]]
  #return X[:, 0] + (X[:, 1])

def total_name(function_transformer, feature_names_in):
  return["sum"]

def household_total_pipeline():
  return make_pipeline(
      imputer,
      FunctionTransformer(household_total, feature_names_out=total_name),
      FunctionTransformer(np.log, feature_names_out="one-to-one"),
      StandardScaler()
  )

#log and standardize pipeline
log_standardize_pipeline = make_pipeline(imputer,
                                        FunctionTransformer(np.log, feature_names_out="one-to-one"),
                                        StandardScaler())

#log1p and standardize pipeline
log1p_standardize_pipeline = make_pipeline(imputer,
                                           FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
                                          StandardScaler())

#polynomial and standardize pipeline
poly_standardize_pipeline = make_pipeline(imputer,
                                          PolynomialFeatures(degree=2),
                                          StandardScaler())

#ratio + 1
def column_ratio_plus_one(X):
  X_np = X
  return X_np[:, [0]] /(X_np[:, [1]] + 1)

def ratio_name(function_transformer, feature_names_in):
  return["ratio"]

def income_per_minor_pipeline():
  return make_pipeline(imputer,
      FunctionTransformer(column_ratio_plus_one, feature_names_out=ratio_name),
      StandardScaler()
  )

#ratio without + 1 to denom
def column_ratio(X):
    X_np = X#.values  # Convert DataFrame to NumPy array
    return X_np[:, [0]] /(X_np[:, [1]])

def ratio_pipeline():
  return make_pipeline(imputer,
      FunctionTransformer(column_ratio, feature_names_out=ratio_name),
  )

def income_per_minor_ploy_standardize_pipeline():
  return make_pipeline(
      imputer,
      FunctionTransformer(column_ratio_plus_one, feature_names_out=ratio_name),
      PolynomialFeatures(degree=2, include_bias=False),
      StandardScaler()
  )

def minors_per_adult_log1p_standardize_pipeline():
  return make_pipeline(
      imputer,
      FunctionTransformer(column_ratio, feature_names_out=ratio_name),
      FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
      StandardScaler()
  )

def income_per_adult_log_standardize_pipeline():
  return make_pipeline(
      imputer,
      FunctionTransformer(column_ratio, feature_names_out=ratio_name),
      FunctionTransformer(np.log, feature_names_out="one-to-one"),
      StandardScaler()
  )

#Combined Pipeline
preprocessing = ColumnTransformer([
    ("standardize", standardize_pipeline, ['RESP_EDU_4_CATEGORIES']),
    ("sqrt", sqrt_standardize_pipeline, ['HH_INCOME']),
    ("onehot", onehot_pipeline, ['HH_COMBINED_PHONE_ACCESS', 'RESP_EDU_4_CATEGORIES', 'HH_ADULTS_COUNT']),
    ("HH_total_size_log", household_total_pipeline(), ['HH_ADULTS_COUNT', 'HH_18_OR_UNDER_COUNT']),
    ("log", log_standardize_pipeline, ['HH_ADULTS_COUNT']),
    ("income_per_minor", income_per_minor_pipeline(), ['HH_INCOME', 'HH_18_OR_UNDER_COUNT']),
    ("income_per_minor_poly", income_per_minor_ploy_standardize_pipeline(), ['HH_INCOME', 'HH_18_OR_UNDER_COUNT']),
    ("log1p", log1p_standardize_pipeline, ['HH_18_OR_UNDER_COUNT']),
    ("minors_per_adult_log1p", minors_per_adult_log1p_standardize_pipeline(), ['HH_18_OR_UNDER_COUNT', 'HH_ADULTS_COUNT']),
    ("income_per_adult_log", income_per_adult_log_standardize_pipeline(), ['HH_INCOME', 'HH_ADULTS_COUNT'])
],
    remainder= 'passthrough')

from sklearn.linear_model import LogisticRegression

final_model = make_pipeline(
    preprocessing,
    LogisticRegression(
        C=0.1,
        class_weight='balanced',
        max_iter=10000,
        penalty='l1',
        solver='saga'
    )
)

final_model.fit(training_features, training_labels)


# In[2]:


def main():
    st.title('Philadelphia Reduced-Price Broadband Internet Program')
    st.write('A data science project by Gibson Hurst.')
    st.header('Apply Here')
    st.write('Answer these questions to apply. Applicants predicted not to have broadband internet access are approved for reduced-priced broadband internet. Applicant screening relies on artificial intelligence and machine learning. Limited to residential customers in Philadelphia, PA. Our mission is to provide broadband internet access to all residents of Philadelphia.')
    
    # Map input options to numeric values
    RESP_EDU_4_CATEGORIES_mapping = {
        "Less than High School": 1,
        "High School Graduate": 2,
        "Some College, No Degree": 3,
        "College Graduate": 4
    }

    HH_INCOME_mapping = {
        "Less than $10,000": 1,
        "$10,000 to $19,999": 2,
        "$20,000 to $29,999": 3,
        "$30,000 to $39,999": 4,
        "$40,000 to $49,999": 5,
        "$50,000 to $74,999": 6,
        "$75,000 to $99,999": 7,
        "$100,000 to $149,999": 8,
        "$150,000 or more": 9,
        "I don't know": None,  # Placeholder for "Don't know"
        "I refuse to answer": None  # Placeholder for "Refused"
    }

    HH_ADULTS_COUNT_mapping = {
        "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6 or more": 6
    }

    HH_18_OR_UNDER_COUNT_mapping = {
        "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10 or more": 10
    }

    HH_COMBINED_PHONE_ACCESS_mapping = {
        "Landline only": 1,
        "Dual with a landline or cell": 2,
        "Cell phone only": 3
    }

    HH_K_12_HOUSEHOLD_mapping = {
        "No": 0, "Yes": 1
    }

    HH_TABLET_mapping = {
        "No": 0, "Yes": 1
    }
    
    AI_mapping = {
        "Opt In": 1, "Opt Out": 0
    }

    # Dropdown menus for user input
    RESP_EDU_4_CATEGORIES = st.selectbox(
        "What is the highest level of school you have completed or the highest degree you have received?",
        options=list(RESP_EDU_4_CATEGORIES_mapping.keys())
    )

    HH_INCOME = st.selectbox(
        "Last year, that is in 2020, what was your total household income from all sources, before taxes?",
        options=list(HH_INCOME_mapping.keys())
    )

    HH_ADULTS_COUNT = st.selectbox(
        "How many adults, age 18 and over, currently live in your household, INCLUDING YOURSELF?",
        options=list(HH_ADULTS_COUNT_mapping.keys())
    )

    HH_18_OR_UNDER_COUNT = st.selectbox(
        "How many people, age less than 18, presently live in your household?",
        options=list(HH_18_OR_UNDER_COUNT_mapping.keys())
    )

    HH_COMBINED_PHONE_ACCESS = st.selectbox(
        "Is your household a landline only household, dual with a landline or cell, or cell phone only household?",
        options=list(HH_COMBINED_PHONE_ACCESS_mapping.keys())
    )

    HH_K_12_HOUSEHOLD = st.selectbox(
        "Are any of the people that live in your household between between the ages of 5 and 18?",
        options=list(HH_K_12_HOUSEHOLD_mapping.keys())
    )

    HH_TABLET = st.selectbox(
        "Do you or any member of your household have a working tablet computer like an iPad, Samsung Galaxy Tab, or Amazon Fire?",
        options=list(HH_TABLET_mapping.keys())
    )
    
    AI = st.selectbox(
        "Do you wish to opt out of AI/ML application screening? AI/ML screening has been shown to expand eligibility for households in need over traditional screening methods",
        options=list(AI_mapping.keys())
    )

    inputs = {
        "RESP_EDU_4_CATEGORIES": RESP_EDU_4_CATEGORIES_mapping[RESP_EDU_4_CATEGORIES],
        "HH_ADULTS_COUNT": HH_ADULTS_COUNT_mapping[HH_ADULTS_COUNT],
        "HH_18_OR_UNDER_COUNT": HH_18_OR_UNDER_COUNT_mapping[HH_18_OR_UNDER_COUNT],
        "HH_COMBINED_PHONE_ACCESS": HH_COMBINED_PHONE_ACCESS_mapping[HH_COMBINED_PHONE_ACCESS],
        "HH_K_12_HOUSEHOLD": HH_K_12_HOUSEHOLD_mapping[HH_K_12_HOUSEHOLD],
        "HH_TABLET": HH_TABLET_mapping[HH_TABLET]
    }
    
    AI = AI_mapping[AI]

    if HH_INCOME == "Don't know":
        inputs["INCOME_IDK"] = 1
        inputs["INCOME_REFUSED"] = 0
        inputs["HH_INCOME"] = np.nan
    elif HH_INCOME == "Refused":
        inputs["INCOME_IDK"] = 0
        inputs["INCOME_REFUSED"] = 1
        inputs["HH_INCOME"] = np.nan
    else:
        inputs["INCOME_IDK"] = 0
        inputs["INCOME_REFUSED"] = 0
        inputs["HH_INCOME"] = HH_INCOME_mapping[HH_INCOME]

    if st.button('Submit Application'):
        features = pd.DataFrame(inputs, index=[0])
        prediction = final_model.predict(features)
        prediction = prediction[0]
        if AI == 1:
            if prediction == 1:
                st.success('Your household has been approved for reduced-price broadband internet')
            else:
                st.warning('Unfortunately, your household does not qualify for reduced-price broadband internet')
        else:
            if inputs['HH_INCOME'] <= 3:
                st.success('Your household has been approved for reduced-price broadband internet')
            else:
                st.warning('Unfortunately, your household does not qualify for reduced-price broadband internet')

    st.header("FAQs")
    st.write("**Q: How is eligibility determined for reduced-price broadband internet?**")
    st.write("A: The machine learning model takes in answers contained in the user's application with the objective of predicting how likely the user's household does not have broadband internet. This information is formatted to allow a mathematical calculation to be applied to the answers. Applicants are approved if their predicted probability of not having broadband internet is greater than a predetermined cutoff value. The mathematical calculation was developed using the machine learning model, which learned from survey data of about 2,500 adults residing in Philadelphia. The survey asked many questions, including those seen on this application and the respondents' household status regarding broadband internet access. The model spotted complex trends in the data that correlate with the status of broadband internet access. These trends are represented in the mathematical calculation, allowing for accurate application screening.")
    st.write("**Q: Why is machine learning used for applicant screening?**")
    st.write("A: In testing, the machine learning model used for applicant screening has been shown to expand eligibility by 42% for households in need over traditional screening methods. In other words, the machine learning model is 42% more accurate than traditional methods at identifying applicants who truly lack broadband internet. Without the machine learning model, applicants would be screened solely based on their household income, limiting any household over a predetermined income threshold from receiving benefits even when needed.")
    st.image('performance1.png')
    st.write("**Q: How were sources of bias addressed in model development?**")
    st.write("A: Addressing and testing for bias in model development was paramount to producing equitable outcomes for applicants. A major step taken to prevent bias was that the model was not trained on protected identifiers such as age, race, gender, zip code, and language. Additionally, the model was trained on a diverse dataset representative of the Philadelphia population. Lastly, the model was optimized to have similar performance identifying both households without broadband internet and household with broadband internet. This was achieved by assessing the model with the balanced accuracy metric, a measure that takes into account the sensitivity and specificity of the model, and by weighting the model for both groups. ")
    st.write("**Q: How consistent is the machine learning model performance across demographics?**")
    st.write("A: Below is a summary comparison of machine learning model performance and income-based screening performance across age, race, and gender. Not enough data was available for performance measurements at the intersection of these demographics.")
    st.image('performace2.png')
    st.write("**Q: What happens if users opt out of having their application reviewed by AI/ML?**")
    st.write("A: Users may opt out of having their application reviewed by AI/ML and will be screened based on their household income in comparison to a predetermined threshold.")

    st.header("About The Project")
    
    st.write("By: Gibson Hurst")
    st.write("[Github Repo](link)")
    st.write("[Experiment Tracking and Versioning](link)")
    st.write("[Data Source](link)")
    st.write("[Portfolio Website](link)")
    st.write("[LinkedIn](link)")
    
    st.subheader("Highlights")
    st.markdown("""
    - fine-tuned a logistic regression model to accurately identify households in need of reduced-price broadband internet
    - achieved greater performace than traditional income-based screening methods for assistance programs, including a 42% increase in sensitivity, an 18% increase in balanced accuracy, and no decrease in specificity
    - logged over 20 experiments with a variety of models and hyperparameters while optimizing for balanced accuracy
    - addressed model bias through feature selection, choice of evaluation metric, balanced class weights, and evaluation across demographics
    - built a web application to show how model implementation would work for programs such as Xfinity Internet Essentials
    """)
    
    st.subheader("Summary")
    st.write("Following the end of the Affordable Connectivity Program, roughly 177,000 low-income households in Philadelphia will see their internet bill increase by $40 due to the end of benefits from this federal program implemented during the pandemic. Sadly, efforts to extend the program fell through in Congress. Currently active public and private internet assistance programs rely on weak income-based heuristics or proxies for low-income such as participation in other assistance programs. In this project, I used data from the Philadelphia 2021 Household Internet Assessment Survey and advanced predictive models to accurately determine households lacking broadband internet while avoiding protected information such as race, gender, age, language, and zip code. My solution can be implemented to screen applicants for internet assistance programs by approving applicants who are predicted to lack broadband internet access in their households. ")
    
    st.subheader("Methodology")
    st.markdown("""
    Task
    * Current internet assistance programs evaluate applicants to determine a need for benefits, using income or a proxy for low-income
    * Hypothesized that other factors besides income, captured in survey data, may be related to the need for reduced-price internet
    * Simplified the task into a binary classification problem using many factors, including income, to predict if a household does or does not have broadband internet
    * Objective: improve sensitivity, which is the accuracy of identifying households without internet, without a loss of specificity, which is the accuracy of identifying households with internet
    * An income-based heuristic is used as the baseline for measuring performance increases 

    Data
    * Sourced data from the 2021 Philadelphia Household Internet Assessment Survey
    * 2,500 households

    Exploratory Data Analysis
    * Training and testing split data was conducted before EDA and data transformations to prevent data leakage
    * Identified 14% of income reports to be unknown or refused to be provided, meaning these non-income values may contain predictive power
    * Identified a strong class imbalance with there being 6x more households with than without broadband internet in the survey data
    * Added calculated features and promising transformations such as log, sqrt, x^2
    * Studied correlations between broadband internet status and original, calculated, and transformed features 

    Data Transformations
    * Performed feature selection and dropped unnecessary features for the task 
    * Build a pipeline for the following transformations, allowing for easy training and inference on new data
    * Duplicated the calculated features that showed promise in the EDA
    * Encoded unknown income and refused to provide income as individual binary features
    * Imputed missing values with training feature median values 
    * Decomposed categorical features with one hot encoding
    * Standardized all continuous features with mean 0 and unit standard deviation

    Modeling 
    * Logged over 20 possible models and countless hyperparameter combinations
    * Experimented with logistic regression and a variety of regularization techniques
    * Experimented with support factor machines and a variety of Kernel tricks
    * Experimented with stochastic gradient descent classifier with a variety of loss functions
    * Experimented with tree-based classifiers such as random forest

    Model Improvements and Evaluation 
    * Created a Google Sheet to store performance metrics and hyperparameters for model comparison and reproducibility
    * Tracked training and cross-validation metrics for all model experiments
    * Performed grid search and random search hyperparameter tuning on all model types 
    * Selected a logistic regression as the final model with C=0.1, balanced class weights, LASSO (l1) regularization to reduce dimensionality, and Saga solver to support the regularization
    * Applied the final model to the testing data and calculated performance metrics 
    * Calculated baseline performance metrics of heuristics such as random chance and household income < $30,000
    * Visualized comparison of performance between the final model, income < $30K heuristic, and Random Chance predictions 

    Addressing Bias
    * Avoided training the model on protected information such as race, age, gender, zip code, language, etc. 
    * Selected balanced accuracy metric for alignment with the goal stated earlier of equal emphasis on sensitivity and specificity
    * Applied class weights to the logistic regression loss function to help resolve the class imbalance and place equal importance on households with and without broadband internet
    * Evaluated model performance across demographics, including race, gender, and age
    * Compared to the same metrics assessed on the income < $30k heuristic

    Final Product
    * Built a web application that simulates an internet assistance program application
    * Transferred the final model into the web application to predict applicants' broadband internet status in their household
    * Added logic to approve any household application that is predicted not to have broadband internet
    * Added an opt-out of ML model evaluation, which will then evaluate the application based on household income
    * Added additional logic to approve any household that opts out of ML model evaluation and has a household income < $30,000
    """
               )
    
    st.write("""
    MIT License

    Copyright (c) 2024 Gibson Hurst

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """)
    
if __name__ == '__main__':
    main()


# In[ ]:




