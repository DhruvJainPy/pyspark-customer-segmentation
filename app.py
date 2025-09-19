import streamlit as st # type: ignore
import pandas as pd # type: ignore
from PIL import Image # type: ignore

# --- Page Configuration ---
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- Caching Function to Load Data ---
@st.cache_data
def load_data(csv_path):
    try:
        return pd.read_csv(csv_path)
    except FileNotFoundError:
        st.error(f"Error: The file '{csv_path}' was not found. Please run the segmentation script first.")
        return None

# --- Main Application ---

# --- Title and Introduction ---
st.title("👥 Customer Segmentation Analysis")
st.markdown("""
This dashboard presents the results of a customer segmentation analysis performed on online retail data.
The customers have been grouped into distinct segments using a Gaussian Mixture Model (GMM) based on their RFM (Recency, Frequency, Monetary) values and other behavioral features.
""")
# --- Load Data and Images ---
DATA_PATH = "output/segmented_customers_gmm.csv/part-00000-6483d9e3-1893-40c2-974e-021f15d2b03d-c000.csv"
PCA_PLOT_PATH = "output/pca_clusters.png"
SILHOUETTE_PLOT_PATH = "output/silhouette_plot.png"

df = load_data(DATA_PATH)

# Stop the app if data loading failed
if df is None:
    st.stop()

# Load images, with error handling
try:
    pca_image = Image.open(PCA_PLOT_PATH)
    silhouette_image = Image.open(SILHOUETTE_PLOT_PATH)
except FileNotFoundError:
    st.warning("Warning: Plot images not found. Please ensure the analysis script has generated them.")
    pca_image, silhouette_image = None, None


# --- Sidebar for User Input ---
st.sidebar.header("Cluster Explorer")
cluster_options = sorted(df['prediction'].unique())
selected_cluster = st.sidebar.selectbox(
    "Select a Cluster to Analyze",
    options=cluster_options,
    format_func=lambda x: f"Cluster {x}"
)

# --- Main Panel for Displaying Results ---

# --- High-Level Model Performance and Visualization ---
st.header("Model Performance and Segment Visualization")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Cluster Distribution (PCA)")
    if pca_image:
        st.image(pca_image, caption="2D PCA projection of customer segments.", width='stretch')
    else:
        st.info("PCA plot image is missing.")

with col2:
    st.subheader("Optimal Number of Clusters")
    if silhouette_image:
        st.image(silhouette_image, caption="Silhouette scores for different numbers of clusters (K).", width='stretch')
    else:
        st.info("Silhouette plot image is missing.")


# --- Cluster Profile Summary ---
st.header("Cluster Profile Summary")
st.markdown("This table shows the average characteristics for each customer segment.")

# Calculate summary stats from the loaded dataframe
profile_cols = ["Recency", "Frequency", "Monetary"]
cluster_profile = df.groupby('prediction')[profile_cols].mean().reset_index()
cluster_profile = cluster_profile.rename(columns={'prediction': 'Cluster'})
cluster_profile['Cluster'] = cluster_profile['Cluster'].apply(lambda x: f"Cluster {x}")

# Add cluster size to the profile
cluster_size = df['prediction'].value_counts().reset_index()
cluster_size.columns = ['prediction', 'NumberOfCustomers']
cluster_profile = cluster_profile.merge(cluster_size, left_on='Cluster', right_on=cluster_size['prediction'].apply(lambda x: f"Cluster {x}")).drop(columns=['prediction'])


st.dataframe(cluster_profile.style.format({
    "Recency": "{:.1f} days",
    "Frequency": "{:.2f}",
    "Monetary": "${:,.2f}"
}), use_container_width=True)


# --- Deep Dive into Selected Cluster ---
st.header(f"Deep Dive into Cluster {selected_cluster}")
st.markdown(f"Exploring the characteristics and a sample of customers from **Cluster {selected_cluster}**.")

# Filter data for the selected cluster
cluster_data = df[df['prediction'] == selected_cluster]

# Display key metrics for the selected cluster
selected_profile = cluster_profile[cluster_profile['Cluster'] == f"Cluster {selected_cluster}"].iloc[0]

st.metric("Number of Customers", f"{selected_profile['NumberOfCustomers']:,}")

col_a, col_b, col_c = st.columns(3)
col_a.metric("Average Recency", f"{selected_profile['Recency']:.1f} days")
col_b.metric("Average Frequency", f"{selected_profile['Frequency']:.2f} purchases")
col_c.metric("Average Monetary Value", f"${selected_profile['Monetary']:,.2f}")

# Display a sample of customers from the selected cluster
st.subheader("Sample Customers")
st.dataframe(cluster_data.head(10), use_container_width=True)

