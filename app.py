import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Load the data
file_path = 'KUESIONER_TB_TOTAL(1) 27 oktober.xlsx'
df = pd.read_excel(file_path)

# Streamlit code for TB data processing
st.title('TB Data Processing Intelligence System')

# Display data tables
st.header('Data Overview')
st.write(df.head())

# Data Preprocessing
st.header('Data Preprocessing')
# Handling missing values
imputer = SimpleImputer(strategy='most_frequent')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
st.write('Missing values handled.')

# Encoding categorical variables
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    df_imputed[column] = label_encoders[column].fit_transform(df_imputed[column])
st.write('Categorical variables encoded.')

# Data Visualization
st.header('Data Visualization')
# Distribution of age
st.subheader('Age Distribution')
fig, ax = plt.subplots()
sns.histplot(df_imputed['Usia'], kde=True, ax=ax)
st.pyplot(fig)

# Save the Streamlit script to a file
streamlit_script = 'tb_questionnaire_streamlit.py'
with open(streamlit_script, 'w') as file:
    file.write(streamlit_code)

print('Streamlit script saved as:', streamlit_script)
