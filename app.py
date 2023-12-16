import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Function to load data
@st.cache
def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        return df
    return None

# Streamlit code for TB data processing and visualization
st.title('TB Data Processing and Visualization')

# Upload file
uploaded_file = st.file_uploader('Choose a file')
if uploaded_file is not None:
    data = load_data(uploaded_file)
    if data is not None:
        st.write('Raw Data')
        st.write(data)
        
        st.header('Data Overview')
        st.write(data.head())
        
        st.header('Data Preprocessing')
        imputer = SimpleImputer(strategy='most_frequent')
        df_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
        
        label_encoders = {}
        for column in df_imputed.select_dtypes(include=['object']).columns:
            label_encoders[column] = LabelEncoder()
            df_imputed[column] = label_encoders[column].fit_transform(df_imputed[column])
        
        st.write('Preprocessed Data')
        st.write(df_imputed.head())
        
        st.header('Data Visualization')
        column_to_plot = st.selectbox('Select a column to visualize', df_imputed.columns)
        fig, ax = plt.subplots()
        sns.histplot(df_imputed[column_to_plot], kde=True, ax=ax)
        ax.set_title('Histogram of ' + column_to_plot)
        st.pyplot(fig)
