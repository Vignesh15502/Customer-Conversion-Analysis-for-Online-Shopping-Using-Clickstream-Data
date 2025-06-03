import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Data Preprocessing
def preprocess_test_data(df):
    # Drop irrelevant columns
    df = df.drop(columns=['year', 'session_id'])

    # Define the mapping from country codes to continent names
    continent_map = {
        1: "Oceania", 2: "Europe", 3: "Europe", 4: "Europe", 5: "Caribbean",
        6: "Oceania", 7: "Europe", 8: "Europe", 9: "Europe", 10: "Europe",
        11: "Europe", 12: "Unknown", 13: "Europe", 14: "Europe", 15: "Europe",
        16: "Europe", 17: "Europe", 18: "Europe", 19: "Europe", 20: "Asia",
        21: "Europe", 22: "Europe", 23: "Europe", 24: "Europe", 25: "Europe",
        26: "North America", 27: "Europe", 28: "Europe", 29: "Europe", 30: "Europe",
        31: "Europe", 32: "Europe", 33: "Europe", 34: "Europe", 35: "Europe",
        36: "Europe", 37: "Europe", 38: "Europe", 39: "Europe", 40: "Asia",
        41: "Europe", 42: "North America", 43: "Unknown", 44: "Unknown",
        45: "Unknown", 46: "Unknown", 47: "Unknown"
    }

    # Map the country codes to continents
    df['continent'] = df['country'].map(continent_map).fillna('Unknown')

    # Define mapping dictionary
    category_map = {
        1: 'trousers',
        2: 'skirts',
        3: 'blouses',
        4: 'sale'
    }

    # Create new column using the mapping
    df['page_1'] = df['page1_main_category'].map(category_map)

    month_name = {4:"April", 5:"May", 6:"June", 7:"July", 8:"August"}
    df["month_names"] = df["month"].map(month_name)

    color_map = {
                    1: "beige",
                    2: "black",
                    3: "blue",
                    4: "brown",
                    5: "burgundy",
                    6: "gray",
                    7: "green",
                    8: "navy blue",
                    9: "of many colors",
                    10: "olive",
                    11: "pink",
                    12: "red",
                    13: "violet",
                    14: "white"
                }

    df['color_name'] = df['colour'].map(color_map)

    location_map = {
                    1: "top left",
                    2: "top in the middle",
                    3: "top right",
                    4: "bottom left",
                    5: "bottom in the middle",
                    6: "bottom right"
                }

    df['location_name'] = df['location'].map(location_map)

    model_photography_map = {
                                1: "en face",
                                2: "profile"
                            }

    df['model_pose'] = df['model_photography'].map(model_photography_map)

    df.drop(columns = ["page1_main_category", "colour", "location", "model_photography"], inplace = True)

    # Initialize the encoder
    label_encoder = LabelEncoder()

    # Apply label encoding
    df['page2_clothing_model_encoded'] = label_encoder.fit_transform(df['page2_clothing_model'])

    # Optionally drop the original column (if not needed anymore)
    df.drop(columns=['page2_clothing_model'], inplace=True)

    # Initialize the encoder
    label_encoder = LabelEncoder()

    # Apply label encoding
    df['continent_encoded'] = label_encoder.fit_transform(df['continent'])

    # Optionally drop the original column (if not needed anymore)
    df.drop(columns=['continent'], inplace=True)
    labelbinarizer = LabelBinarizer()

    encoded_results_2 = labelbinarizer.fit_transform(df["page_1"])
    df_encoded_2 = pd.DataFrame(encoded_results_2,columns=labelbinarizer.classes_)

    encoded_results_3 = labelbinarizer.fit_transform(df["month_names"])
    df_encoded_3 = pd.DataFrame(encoded_results_3,columns=labelbinarizer.classes_)

    encoded_results_4 = labelbinarizer.fit_transform(df["color_name"])
    df_encoded_4 = pd.DataFrame(encoded_results_4,columns=labelbinarizer.classes_)

    encoded_results_5 = labelbinarizer.fit_transform(df["location_name"])
    df_encoded_5 = pd.DataFrame(encoded_results_5,columns=labelbinarizer.classes_)

    df = pd.concat([df,df_encoded_2, df_encoded_3, df_encoded_4, df_encoded_5], axis=1)

    # Initialize the encoder
    label_encoder = LabelEncoder()

    # Apply label encoding
    df['model_pose_encoded'] = label_encoder.fit_transform(df['model_pose'])

    # Optionally drop the original column (if not needed anymore)
    df.drop(columns=['model_pose'], inplace=True)

    df.drop(columns = ["page_1","month_names","color_name","location_name"], inplace = True)

    print(df.columns)
    print(df.columns.size)
    return df


# Load your trained models
with open(r"D:\VIGNESH P\Nithya projects\Click stream data\models\classification_model.pkl", "rb") as rmc:
    classification_model = pickle.load(rmc)

with open(r"D:\VIGNESH P\Nithya projects\Click stream data\models\regression_model.pkl", "rb") as rmr:
    regression_model = pickle.load(rmr)

# App Title
st.set_page_config(page_title="Customer Conversion Analysis", layout="wide")
st.title("🛒 Customer Conversion Analysis using Clickstream Data")

# Sidebar
st.sidebar.header("Navigation")
section = st.sidebar.radio("Go to", ["Upload Data", "Customer Conversion Prediction", "Revenue Estimation", "Customer Segmentation"])

# Upload Data
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    return df

# Section: Upload Data
if section == "Upload Data":
    st.header("📂 Upload Clickstream Data")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        data = load_data(uploaded_file)
        st.success("Data Loaded Successfully!")
        st.dataframe(data)

# Section: Customer Conversion Prediction
elif section == "Customer Conversion Prediction":
    st.header("🎯 Predict Customer Conversion")

    uploaded_file = st.file_uploader("Upload Test CSV", type=["csv"], key="convert")
    if uploaded_file:
        df = load_data(uploaded_file)

        # Preprocessing steps
        df_processed = preprocess_test_data(df)
        df_processed["price_2"] = df_processed["price_2"].replace({2:0})

        # For now, assume preprocessed

        st.write("✅ Sample of input data:")
        st.dataframe(df)

        st.write("✅ Preprocessed Data:")
        st.dataframe(df_processed)

        if st.button("Predict Conversion"):
            predictions = classification_model.predict(df_processed.drop("price_2", axis = 1))
            df['Predicted Conversion'] = predictions
            st.success("Prediction complete.")
            df["price_2"] = df["price_2"].replace({2:0})
            st.dataframe(df[['price_2', 'Predicted Conversion']])
            st.download_button("Download Results", df.to_csv(index=False), "predicted_conversion.csv")

# Section: Revenue Estimation
elif section == "Revenue Estimation":
    st.header("💰 Estimate Customer Revenue")

    uploaded_file = st.file_uploader("Upload Test CSV", type=["csv"], key="revenue")
    if uploaded_file:
        df = load_data(uploaded_file)

        #Orginal Data
        st.write("✅ Sample of input data:")
        st.dataframe(df.head())

        # Preprocessing steps
        df_processed = preprocess_test_data(df)
        st.write("✅ Preprocessed Data:")
        st.dataframe(df_processed.head())

        if st.button("Estimate Revenue"):
            predictions = regression_model.predict(df_processed.drop("price", axis = 1))
            df['Estimated Revenue'] = predictions
            st.success("Revenue estimation complete.")
            st.dataframe(df[['price','Estimated Revenue']])
            st.download_button("Download Results", df.to_csv(index=False), "estimated_revenue.csv")

# Section: Customer Segmentation (for future use)
elif section == "Customer Segmentation":
    st.header("👥 Customer Segmentation (Clustering)")

    # Upload Dataset
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### 📊 Raw Data")
        st.dataframe(df)

        # Preprocessing steps
        df_processed = preprocess_test_data(df)
        st.write("✅ Preprocessed Data:")
        st.dataframe(df_processed.head())

        # Select features
        selected_features = st.multiselect("Select features for clustering", df_processed.columns.tolist())
        
        if selected_features:
            X = df_processed[selected_features]

            # Standardization
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Elbow Method
            st.subheader("2. Elbow Method - Optimal K")
            inertia = []
            K = range(1, 11)
            for k in K:
                km = KMeans(n_clusters=k, random_state=42)
                km.fit(X_scaled)
                inertia.append(km.inertia_)

            fig1, ax1 = plt.subplots()
            ax1.plot(K, inertia, marker='o')
            ax1.set_xlabel("Number of Clusters")
            ax1.set_ylabel("Inertia")
            ax1.set_title("Elbow Method")
            st.pyplot(fig1)

            # Choose K
            k_value = st.slider("Choose number of clusters (K)", 2, 10, 4)
            
            # Apply KMeans
            st.subheader("3. KMeans Clustering")
            model = KMeans(n_clusters=k_value, random_state=42)
            cluster_labels = model.fit_predict(X_scaled)
            df['Cluster'] = cluster_labels

            st.write("### 🧾 Clustered Data")
            st.dataframe(df)

            # PCA for Visualization
            pca = PCA(n_components=2)
            pca_data = pca.fit_transform(X_scaled)

            st.subheader("4. Visualize Clusters (PCA Reduced)")
            fig2, ax2 = plt.subplots()
            scatter = ax2.scatter(pca_data[:, 0], pca_data[:, 1], c=cluster_labels, cmap='tab10', s=10)
            ax2.set_xlabel("PC1")
            ax2.set_ylabel("PC2")
            ax2.set_title("Customer Segments (PCA Reduced)")
            legend_labels = np.unique(cluster_labels)
            ax2.legend(handles=scatter.legend_elements()[0], labels=[str(i) for i in legend_labels], title="Cluster")
            st.pyplot(fig2)


