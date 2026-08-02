# 🫁 Lung Cancer Prediction App

An interactive machine learning web application built with **Python**, **Scikit-Learn**, and **Streamlit** to predict the risk of lung cancer based on patient survey responses.

---

## 🚀 Features
- **Machine Learning Model:** Uses a `RandomForestClassifier` with balanced class weights to handle dataset imbalances.
- **Data Scaling:** Automatically scales numerical features (like age) using `MinMaxScaler`.
- **Interactive UI:** Built with Streamlit, replacing traditional terminal text inputs with user-friendly checkboxes and number selectors to prevent user input errors.
- **Model Evaluation:** Displays the Cross-Validation Score, Confusion Matrix, Classification Report, and Accuracy Score right in the app.

---

## 🛠️ Prerequisites & Installation

Make sure you have Python installed, then clone the repository and install the required dependencies:

```bash
# Clone the repository
git clone [https://github.com/RajveerRaushanclasssix/Lung-cancer-detection-model.git](https://github.com/RajveerRaushanclassix/Lung-cancer-prediction-model.git)
cd Lung-cancer-detection-model

# Install required libraries
pip install pandas numpy scikit-learn streamlit matplotlib
