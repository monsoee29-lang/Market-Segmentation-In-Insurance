# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle

st.set_page_config(page_title="Credit Card Customer Segmentation", layout="wide")
st.title("💳 Credit Card Customer Segmentation")

#Sidebar Controls
st.sidebar.header("Parameter Interface")
x_feature = st.sidebar.selectbox("X-axis for Visualization", ["PCA1", "PCA2"], index=0)
y_feature = st.sidebar.selectbox("Y-axis for Visualization", ["PCA1", "PCA2"], index=1)
num_clusters = st.sidebar.slider("Number of clusters", 2, 6, 3)

#Load scaler and dataset
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
df = pd.read_csv("Customer Data.csv")

if "CUST_ID" in df.columns:
    df.drop(columns=["CUST_ID"], inplace=True)

#Fill missing values
df["MINIMUM_PAYMENTS"].fillna(df["MINIMUM_PAYMENTS"].mean(), inplace=True)
df["CREDIT_LIMIT"].fillna(df["CREDIT_LIMIT"].mean(), inplace=True)

#Feature Engineering
df_grouped = pd.DataFrame({
    "Balance_Info": df["BALANCE"] * df["BALANCE_FREQUENCY"],
    "Purchases": df["PURCHASES"] + df["ONEOFF_PURCHASES"] + df["INSTALLMENTS_PURCHASES"],
    "Purchases_Frequency": df["PURCHASES_FREQUENCY"] + df["ONEOFF_PURCHASES_FREQUENCY"] + df["PURCHASES_INSTALLMENTS_FREQUENCY"],
    "Purchases_TRX": df["PURCHASES_TRX"],
    "Cash_Advance": df["CASH_ADVANCE"],
    "Cash_Advance_Frequency": df["CASH_ADVANCE_FREQUENCY"],
    "Cash_Advance_TRX": df["CASH_ADVANCE_TRX"],
    "Credit_Info": df["CREDIT_LIMIT"] + df["PAYMENTS"] + df["MINIMUM_PAYMENTS"],
    "Full_Payment_Ratio": df["PRC_FULL_PAYMENT"],
    "Tenure": df["TENURE"]
}).fillna(0)

#Scale Data
X_scaled = scaler.transform(df_grouped)

#KMeans Clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
df_grouped["Cluster"] = kmeans.fit_predict(X_scaled)

#Cluster Naming (I defined names based on domain knowledge)
default_names = [
    "Low Usage Customers",
    "High-Value Active Spenders",
    "Cash Advance Heavy Users",
    "Balanced Regular Customers",
    "New/Low Tenure Customers",
    "Premium Loyal Customers"
]

cluster_names = {i: default_names[i] for i in range(num_clusters)}

df_grouped["Segment"] = df_grouped["Cluster"].map(cluster_names)

#PCA for Visualization
pca = PCA(n_components=2)
pca_data = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(pca_data, columns=["PCA1", "PCA2"])
pca_df["Segment"] = df_grouped["Segment"]
pca_df["Cluster"] = df_grouped["Cluster"]
centroids_pca = pca.transform(kmeans.cluster_centers_)

#Visualization
st.subheader("📊 Cluster Visualization (PCA)")

fig, ax = plt.subplots(figsize=(8, 6))
for c in range(num_clusters):
    subset = pca_df[pca_df["Cluster"] == c]
    ax.scatter(
        subset[x_feature],
        subset[y_feature],
        label=cluster_names[c],
        alpha=0.6)

ax.scatter(
    centroids_pca[:, 0],
    centroids_pca[:, 1],
    c="red",
    s=200,
    marker="X",
    label="Centroids")

ax.set_xlabel(x_feature)
ax.set_ylabel(y_feature)
ax.legend()
st.pyplot(fig)

#Cluster Profiles
st.subheader("📈 Cluster Profiles (Average Behavior)")
profile_df = df_grouped.groupby("Segment").mean().round(2)
st.dataframe(profile_df)

#Data Table
st.subheader("📋 Customer Data by Segment")
segment_choice = st.sidebar.selectbox(
    "Select Segment",
    list(cluster_names.values())
)

st.dataframe(df_grouped[df_grouped["Segment"] == segment_choice])

#Prediction Section
st.markdown("---")
st.header("🔮 Predict Segment for New Customer")

new_inputs = {}
cols = st.columns(2)

for i, feature in enumerate(df_grouped.columns.drop(["Cluster", "Segment"])):
    col = cols[i % 2]
    new_inputs[feature] = col.number_input(
        feature,
        float(df_grouped[feature].min()),
        float(df_grouped[feature].max()),
        float(df_grouped[feature].mean())
    )

if st.button("Predict Segment"):
    new_point = np.array([list(new_inputs.values())])
    new_point_scaled = scaler.transform(new_point)
    pred_cluster = kmeans.predict(new_point_scaled)[0]
    pred_segment = cluster_names[pred_cluster]

    st.success(
        f"✅ New customer belongs to **{pred_segment}** (Cluster {pred_cluster})")
