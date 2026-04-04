🚀 Predictive Customer Segmentation Dashboard
Developed by Ashutosh Tripathi
An end-to-end Machine Learning application that transforms raw customer data into actionable marketing intelligence. This project utilizes Unsupervised Learning (GMM) and Principal Component Analysis (PCA) to classify customers into distinct behavioral personas.

📌 Project Overview
Traditional marketing often treats all customers the same. This project solves that problem by:

Analyzing complex spending patterns and income levels.

Segmenting customers into 4 data-driven personas.

Prescribing specific marketing strategies for each segment to maximize ROI.

🛠️ The Tech Stack
Language: Python 3.x

ML Frameworks: Scikit-Learn (PCA, Gaussian Mixture Models)

Dashboard: Streamlit

Visualizations: Plotly (Radar Charts, Longitudinal Analysis)

Data Handling: Pandas, NumPy

Model Deployment: Joblib

🧬 Data Science Workflow
As seen in the Customer Segmentation Notebook, the pipeline follows:

Feature Engineering: Calculated total spend and handled parental status.

Preprocessing: Log-transformation to handle skewed distributions and Standard Scaling.

Dimensionality Reduction: Used PCA to reduce noise and visualize clusters.

Clustering: Implemented Gaussian Mixture Models (GMM) for soft-clustering and better segment boundaries compared to K-Means.

🎯 Key Customer Personas
🥂 High-Value Elite: Top-tier spenders focused on premium products like Wines.

🛒 Value-Conscious Family: High-volume shoppers looking for deals and essentials.

🏠 Established Household: Mid-to-high income stable spenders interested in convenience.

🌱 Emerging Minimalist: New or younger customers with low current spend but high growth potential.

🚀 How to Run Locally
Clone the repository:

Bash
git clone https://github.com/your-username/customer-segmentation-dashboard.git
cd customer-segmentation-dashboard
Install dependencies:

Bash
pip install -r requirements.txt
Launch the Dashboard:

Bash
streamlit run app.py
📁 Project Structure
Plaintext
├── app.py                # Streamlit Deployment Script
├── requirements.txt      # Production Dependencies
├── models/               # Pre-trained Scaler, PCA, and GMM models
│   ├── scaler.pkl
│   ├── pca.pkl
│   └── gmm_model.pkl
└── Notebook/             # Research & Development
    └── Customer Segmentation.ipynb
 
Prerequisites

Ensure you have Python installed. Then, install the dependencies:
```bash
pip install -r requirements.txt

 
License: MIT

Author: Ashutosh Tripathi