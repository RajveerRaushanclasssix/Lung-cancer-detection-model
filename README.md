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

Make sure you have Python installed on your system.

### 1. Clone the Repository
Open your terminal or command prompt and run:

```bash
git clone https://github.com/RajveerRaushanclasssix/Lung-cancer-detection-model
cd Lung-cancer-detection-model
```
### 2. Install the prequisits
```bash
pip install numpy pandas scikit-learn streamlit imbalanced-learn
```
### 3. Run the main.py file
```bash
streamlit run main.py
```


