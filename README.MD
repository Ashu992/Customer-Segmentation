# 📊 Customer Behavioral Analytics & Segmentation
**Developed by Ashutosh Tripathi** *Strategic Data Science Solution for Targeted Marketing*

---

## 📑 Executive Summary
This project delivers an end-to-end analytical pipeline that transforms raw customer transaction data into **actionable marketing intelligence**. By leveraging **Unsupervised Machine Learning**, we move beyond simple demographics to understand the *psychographics* of purchasing behavior.

The core solution features a **Gaussian Mixture Model (GMM)** clustering engine integrated into a high-performance **Streamlit Dashboard** for real-time customer persona classification.

---

## 🛠️ Technical Architecture
| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Core logic and processing |
| **Analysis** | Scikit-Learn, PCA | Dimensionality reduction & GMM Clustering |
| **Interface** | Streamlit | Real-time interactive UI |
| **Visualization** | Plotly | Dynamic Radar Charts & Longitudinal tracking |
| **Persistence** | Joblib | Serialization of pre-trained ML pipelines |

---

## 🧬 Analytical Workflow
1. **Feature Engineering:** Synthesized "Total Spend" and "Parental Status" to identify household economic pressure.
2. **Preprocessing:** Applied Log-Transformation to normalize skewed financial data and `StandardScaler` for feature parity.
3. **Dimensionality Reduction:** Utilized **PCA** to condense features while retaining 80%+ variance, eliminating noise for better cluster separation.
4. **Clustering Engine:** Implemented **Gaussian Mixture Models (GMM)** to allow for "soft clustering," providing a more nuanced segment boundary than traditional K-Means.

---

## 🎯 Business Personas & Strategy
The model identifies four distinct segments, each requiring a unique marketing approach:

* **🥂 High-Value Elite:** Top-tier income; focus on premium wines and luxury meat products.  
    *Strategy:* Exclusive VIP events and early-access previews.
* **🛒 Value-Conscious Family:** High-volume essentials; sensitive to price per unit.  
    *Strategy:* Bulk-buy discounts and essential pantry subscription bundles.
* **🏠 Established Household:** Stable mid-to-high income; values convenience.  
    *Strategy:* Time-saving meal kits and automated recurring orders.
* **🌱 Emerging Minimalist:** Lower current spend but high growth potential.  
    *Strategy:* First-purchase incentives and social-media-driven flash sales.

---

## 🚀 Deployment & Installation

### Prerequisites
* Python 3.8 or higher
* Validated `requirements.txt` dependencies

### Local Setup
```bash
# 1. Clone the repository
git clone [https://github.com/Ashu992/Customer-Segmentation.git](https://github.com/Ashu992/Customer-Segmentation.git)
cd Customer-Segmentation

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Dashboard
streamlit run app.py

## Project Structure

├── app.py                # Streamlit Production Dashboard
├── requirements.txt      # Environment Dependencies
├── models/               # Serialized ML Pipelines
│   ├── scaler.pkl        # Data Normalizer
│   ├── pca.pkl           # Dimensionality Reducer
│   └── gmm_model.pkl     # Primary Clustering Model
└── Notebook/             # Research & Development
    └── Customer Segmentation.ipynb

License: MIT | Author: Ashutosh Tripathi
