import streamlit as st
import pandas as pd

# Function to load data
@st.cache
def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        return df
    return None

# Streamlit code to upload data and display the data table
st.title('TB Questionnaire Data Upload')

uploaded_file = st.file_uploader('Choose a file')
if uploaded_file is not None:
    data = load_data(uploaded_file)
    if data is not None:
        st.write(data)

# Save the Streamlit script to a Python file
streamlit_script = """
import streamlit as st
import pandas as pd

@st.cache
def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        return df
    return None

st.title('TB Questionnaire Data Upload')

uploaded_file = st.file_uploader('Choose a file')
if uploaded_file is not None:
    data = load_data(uploaded_file)
    if data is not None:
        st.write(data)
"""

with open('tb_questionnaire_streamlit.py', 'w') as file:
    file.write(streamlit_script)

print('Streamlit script saved as tb_questionnaire_streamlit.py')
# visualisasi
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Function to load data
@st.cache
def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        return df
    return None

# Streamlit code to upload data and display the data table
st.title('Data Visualization with Streamlit')

uploaded_file = st.file_uploader('Choose a file')
if uploaded_file is not None:
    data = load_data(uploaded_file)
    if data is not None:
        st.write('Raw Data')
        st.write(data)
        
        # Select a column to visualize
        column_to_plot = st.selectbox('Select a column to visualize', data.columns)
        
        # Create a histogram
        fig, ax = plt.subplots()
        data[column_to_plot].hist(ax=ax)
        ax.set_title('Histogram of ' + column_to_plot)
        st.pyplot(fig)
