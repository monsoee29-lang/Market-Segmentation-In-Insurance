# Credit Card Customer Segmentation (K-Means Clustering)
Developed by: Ei Mon Soe

Major: Statistics and Data Science (Parami University)

# Quick Links
🚀 Live Web Application: https://market-segmentation-in-insurance-eimon.streamlit.app/

📂 Dataset Source: https://www.kaggle.com/datasets/jillanisofttech/market-segmentation-in-insurance-unsupervised/data

📌 Project Overview
This project applies Unsupervised Machine Learning to segment credit card users into distinct behavioral groups. By analyzing spending habits, credit usage, and payment patterns, this model helps financial institutions understand customer profiles to create targeted marketing strategies and improve risk management.

📊 Data Science Workflow
1. Data Cleaning & Preprocessing
Credit card datasets often contain missing values in financial fields. I handled these using:

Mean Imputation: Applied to MINIMUM_PAYMENTS and CREDIT_LIMIT to maintain the integrity of the distribution.

Feature Removal: Dropped CUST_ID as it contains no predictive value.

Correlation Analysis: Used Seaborn heatmaps to identify multicollinearity between purchase frequencies and installment types.

2. Strategic Feature Engineering
To reduce noise and improve interpretability, I grouped the original variables into 6 behavioral categories:

Balance Info: Weighted balance based on frequency of updates.

Purchases: Combined one-off and installment purchases into a total volume and frequency index.

Cash Advance: Isolated cash-heavy transactions.

Credit & Payments: Aggregated credit limits and total payment history.

Full Payment Ratio: Captured the percentage of debt cleared monthly.

Tenure: Accounted for customer loyalty/duration.

3. Clustering & Optimization
Scaling: Applied StandardScaler to ensure all financial metrics contribute equally to the distance calculation.

Optimal K Selection: Used a combination of the Elbow Method (WCSS) and Silhouette Analysis.

Decision Logic: While K=2 had a higher silhouette score, K=3 was selected to provide better business utility—separating customers into Low, Medium, and High Spenders.

Dimensionality Reduction (PCA): Implemented Principal Component Analysis (PCA) to visualize the 10-dimensional feature space in 2D, confirming clear separation between clusters.

🚀 Tech Stack
Language: Python

Libraries: Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib

Deployment: Streamlit Cloud (Expected)

Model Persistence: Pickle (Model and Scaler)

📈 Key Insights
Cluster 0 (Low Spenders): Customers who primarily use the card for small, infrequent purchases with low balances.

Cluster 1 (Cash-Heavy Users): Customers with high cash advance usage and lower purchase frequencies.

Cluster 2 (High-Value Spenders): Customers with high credit limits and frequent, large purchases.
