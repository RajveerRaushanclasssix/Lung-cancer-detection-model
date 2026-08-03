import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

st.set_page_config(page_title="Lung Cancer Prediction", layout="wide")

# 1. Load & Preprocess Data (Cached to run only once)
@st.cache_data
def load_data():
    dataset = pd.read_csv("survey lung cancer.csv")
    dataset.columns = dataset.columns.str.strip()

    dataset["GENDER"] = np.where(dataset["GENDER"] == "M", 0, 1)

    binary_cols = [
        "SMOKING", "YELLOW_FINGERS", "ANXIETY", "PEER_PRESSURE",
        "CHRONIC DISEASE", "FATIGUE", "ALLERGY", "WHEEZING",
        "ALCOHOL CONSUMING", "COUGHING", "SHORTNESS OF BREATH",
        "SWALLOWING DIFFICULTY", "CHEST PAIN"
    ]
    for col in binary_cols:
        dataset[col] = np.where(dataset[col] == 2, 1, 0)

    dataset["LUNG_CANCER"] = np.where(dataset["LUNG_CANCER"] == "YES", 1, 0)
    return dataset

@st.cache_resource
def load_eval_model():

    dataset = load_data()

    y = dataset["LUNG_CANCER"].values
    x = dataset.drop(columns=["LUNG_CANCER"]).values

    # 2. Train / Test Split
    train_x, test_x, train_y, test_y = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Fit Pipeline Properly
    scaler = MinMaxScaler()
    train_x_scaled = scaler.fit_transform(train_x)
    test_x_scaled = scaler.transform(test_x)

    smote = SMOTE(random_state=42)
    x_train_resampled, y_train_resampled = smote.fit_resample(train_x_scaled, train_y)

    # Model Training without duplicate class weighting
    model = LogisticRegression(class_weight="balanced")
    model.fit(x_train_resampled, y_train_resampled)

    # 4. Correct Out-of-Fold Cross Validation
    cv_pipeline = ImbPipeline([
        ('scaler', MinMaxScaler()),
        ('smote', SMOTE(random_state=42)),
        ('model', LogisticRegression(class_weight='balanced',random_state=42))
    ])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(cv_pipeline, x, y, cv=skf, scoring="accuracy")

    preds = model.predict(test_x_scaled)

    acc = accuracy_score(preds , test_y)
    CM = confusion_matrix(preds , test_y)
    CR = classification_report(preds , test_y)

    return model , scaler , acc , CM, CR , cv_scores

model , scaler , acc , confusion_matrix , lassification_report , CVS = load_eval_model()

# 5. Model Evaluation 

st.subheader("===== Model Clarifications ======")

st.write(f"Cross validation score : {np.round(CVS * 100, 2)}")

st.write("Confusion Matrix:")
st.code(confusion_matrix)  # FIXED ORDER

report = classification_report  # FIXED ORDER
st.write("Classification Report:")
st.code(report)

acc = acc * 100  # FIXED ORDER
st.write(f"Accuracy : {acc:.2f}%")

# 6. Form Interface
st.subheader("===== Patient Survey Form =====")

f1 = st.selectbox("What is your gender", ("Male", "Female"))
f1 = 0 if f1 == "Male" else 1

age = st.number_input("Enter your age : ", min_value=0, max_value=120, value=30)

f2 = 1 if st.checkbox("Do you smoke?") else 0
f3 = 1 if st.checkbox("Do you have yellow fingers?") else 0
f4 = 1 if st.checkbox("Do you have anxiety?") else 0
f5 = 1 if st.checkbox("Do you experience peer pressure?") else 0
f6 = 1 if st.checkbox("Do you have a chronic disease?") else 0
f7 = 1 if st.checkbox("Do you experience fatigue?") else 0
f8 = 1 if st.checkbox("Do you have allergies?") else 0
f9 = 1 if st.checkbox("Do you experience wheezing?") else 0
f10 = 1 if st.checkbox("Do you consume alcohol?") else 0    
f11 = 1 if st.checkbox("Do you have coughing?") else 0
f12 = 1 if st.checkbox("Do you experience shortness of breath?") else 0
f13 = 1 if st.checkbox("Do you have swallowing difficulty?") else 0
f14 = 1 if st.checkbox("Do you have chest pain?") else 0

if st.button("Predict"):
    DSet = np.array([[f1, age, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14]], dtype=float)
    DSet_scaled = scaler.transform(DSet)

    risk_prob = model.predict_proba(DSet_scaled)[0][1]

    st.markdown("---")
    st.subheader("Results")
    if risk_prob >= 0.35:
        st.error(f"Yes, model predicted that you have a risk of lung cancer. (Probability: {risk_prob:.1%})")
    else:
        st.success(f"No, model predicted that you don't have lung cancer. (Probability: {risk_prob:.1%})")
    st.warning("Disclaimer: Not every prediction is accurate. For serious health issues, please consult a real doctor.")
