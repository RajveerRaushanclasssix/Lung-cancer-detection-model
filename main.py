import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix , classification_report , accuracy_score


dataset = pd.read_csv("survey lung cancer.csv")
dataset.columns = dataset.columns.str.strip()

# Converting the categorical data to numerical data

dataset["GENDER"] = np.where(dataset["GENDER"] =="M" , 0 ,1)
dataset["SMOKING"] = np.where(dataset["SMOKING"] == 2 , 1  , 0)
dataset["YELLOW_FINGERS"] = np.where(dataset["YELLOW_FINGERS"] == 2 , 1  , 0)
dataset["ANXIETY"] = np.where(dataset["ANXIETY"] == 2 , 1  , 0)
dataset["PEER_PRESSURE"] = np.where(dataset["PEER_PRESSURE"] == 2 , 1  , 0)
dataset["CHRONIC DISEASE"] = np.where(dataset["CHRONIC DISEASE"] == 2 , 1  , 0)
dataset["FATIGUE"] = np.where(dataset["FATIGUE"] == 2 , 1  , 0)
dataset["ALLERGY"] = np.where(dataset["ALLERGY"] == 2 , 1  , 0)
dataset["WHEEZING"] = np.where(dataset["WHEEZING"] == 2 , 1  , 0)
dataset["ALCOHOL CONSUMING"] = np.where(dataset["ALCOHOL CONSUMING"] == 2 , 1  , 0)
dataset["COUGHING"] = np.where(dataset["COUGHING"] == 2 , 1  , 0)
dataset["SHORTNESS OF BREATH"] = np.where(dataset["SHORTNESS OF BREATH"] == 2 , 1  , 0)
dataset["SWALLOWING DIFFICULTY"] = np.where(dataset["SWALLOWING DIFFICULTY"] == 2 , 1  , 0)
dataset["CHEST PAIN"] = np.where(dataset["CHEST PAIN"] == 2 , 1  , 0)
dataset["LUNG_CANCER"] = np.where(dataset["LUNG_CANCER"] == "YES" , 1  , 0)

# Splitting the dataset into training and testing sets and scaling the data

y = np.array(dataset["LUNG_CANCER"] , dtype=float)
x = np.array(dataset.drop(columns=["LUNG_CANCER"]), dtype=float)

train_x , test_x , train_y , test_y = train_test_split(x, y, test_size=0.2, random_state=42)
scaler = MinMaxScaler()

train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)

# Checking the cross_val_score of the model

model = RandomForestClassifier(n_estimators=100 , class_weight = 'balanced' ,random_state=42)
cv_score = cross_val_score(model , x , y , cv=5 , scoring="accuracy")

st.subheader("===== Model clarifications ======")

st.write(f"Cross validation score : {cv_score * 100}")

# Training the model

model.fit(train_x, train_y)

# Testing the model

preds = model.predict(test_x)

st.write(f'''Confusion matrix : " 
 {confusion_matrix(preds , test_y)}
 ''')

report = classification_report(preds , test_y)

st.write(f'''Classification_report : 

 {report}

 ''')
st.write(f'''Accuracy :
 {int(accuracy_score(preds , test_y) * 100)} "%"
 ''')



st.subheader("===== Patient survey form =====")

f1 = st.selectbox("What is your gender",("Male" , "Female"))
f1 = 0 if f1 == "Male" else 1

age = st.number_input("Enter your age : " , min_value=0 , max_value=120 , value=30)

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

classes = {0:"No , you don't have lung cancer." , 1:"Yes , model predicted that you have risk of lung cancer."}

if st.button("Predict"):
    DSet = np.array([[f1, age ,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14]] , dtype=float)
    DSet = scaler.transform(DSet)

    pred = model.predict(DSet)

    st.markdown("---")
    st.subheader("Results")
    st.write(f"**Prediction:** {classes.get(int(pred[0]))}")
    st.warning("Disclaimer: Not every prediction is accurate. For serious health issues, please consult a real doctor.")
