import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Function to load data
@st.cache
def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        return df
    return None

# Streamlit code for TB data processing, visualization, PCA analysis, and clustering
st.title('TB Data Processing, Visualization, PCA Analysis, and Clustering')

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
        
        st.header('PCA Analysis')
        numerical_data = df_imputed.select_dtypes(include=['float64', 'int64'])
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numerical_data)
        
        num_components = st.slider('Select the number of PCA components', min_value=2, max_value=min(len(numerical_data.columns), 10), value=2)
        pca = PCA(n_components=num_components)
        pca_result = pca.fit_transform(scaled_data)
        
        pca_columns = [f'PC{i+1}' for i in range(num_components)]
        pca_df = pd.DataFrame(data=pca_result, columns=pca_columns)
        st.write('PCA Result')
        st.write(pca_df.head())
        
        st.subheader('PCA Visualization Options')
        visualization_option = st.radio("Select PCA Visualization Type", ('Scatter Plot', 'Heatmap'))
        
        # Visualize PCA result based on user choice
        st.subheader('PCA Visualization')
        if visualization_option == 'Scatter Plot':
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=pca_df.columns[0], y=pca_df.columns[1], data=pca_df)
            plt.title('PCA Scatter Plot')
            st.pyplot(plt)
        elif visualization_option == 'Heatmap':
            plt.figure(figsize=(10, 8))
            sns.heatmap(pca_df.corr(), annot=True, cmap='coolwarm')
            plt.title('PCA Correlation Heatmap')
            st.pyplot(plt)
        
        st.header('Clustering')
        num_clusters = st.slider('Select the number of clusters for KMeans', min_value=2, max_value=10, value=3)
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(pca_df)
        
        cluster_labels = kmeans.labels_
        pca_df['Cluster'] = cluster_labels
        
        st.write('Clustering Results')
        st.write(pca_df.head())
        
        if st.checkbox('Show Cluster Visualization'):
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=pca_df.columns[0], y=pca_df.columns[1], hue='Cluster', data=pca_df, palette='viridis')
            plt.title('Cluster Visualization')
            st.pyplot(plt)
        
        if st.checkbox('Label Clusters'):
            cluster_labels_input = st.text_input('Enter labels for clusters separated by commas (e.g., "Cluster 1,Cluster 2,Cluster 3")')
            cluster_labels_list = [label.strip() for label in cluster_labels_input.split(',')]
            
            if len(cluster_labels_list) == num_clusters:
                pca_df['Cluster'] = pca_df['Cluster'].map({i: label for i, label in enumerate(cluster_labels_list)})
                st.write('Cluster labels applied:')
                st.write(pca_df.head())
            else:
                st.warning(f'Please provide {num_clusters} labels for each cluster.')
