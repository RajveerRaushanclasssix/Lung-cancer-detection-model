import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
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

# Visualizing the data
'''
x = x[:,0]
plt.scatter(x, y)
plt.show()
'''
# Checking the cross_val_score of the model

model = RandomForestClassifier(n_estimators=100 , class_weight = 'balanced' ,random_state=42)
cv_score = cross_val_score(model , x , y , cv=5 , scoring="accuracy")
print(f'''\n

=================================== Model clarifications ========================================

Cross value score : {cv_score * 100}

''')

# Training the model

model.fit(train_x, train_y)

# Testing the model

preds = model.predict(test_x)

print(f'''Confusion matrix : " 

 {confusion_matrix(preds , test_y)}
 
 ''')
print(f'''Classification_report : 

 {classification_report(preds , test_y)}
 
 ''')
print(f'''Accuracy :

 {int(accuracy_score(preds , test_y) * 100)} "%"
 
 ''')


def take_data():
    f1 = input("What is your gender (M/F) : ").lower()
    if f1 == 'm':
        f1 = 0
    else :
        f1 =1

    age = input("Enter your age : ")

    f2 = input("Do you smoke Y/N : ").lower()
    if f2 == 'y':
        f2=1
    else:
        f2=0

    f3 = input("Do you have yellow fingers Y/N : ").lower()
    if f3 == 'y':
        f3=1
    else:
        f3=0

    f4 = input("Do you have anxiety Y/N : ").lower()
    if f4 == 'y':
        f4=1
    else:
        f4=0

    f5 = input("Do you have peer pressure Y/N : ").lower()
    if f5 == 'y':
        f5=1
    else:
        f5=0

    f6 = input("Do you have chronic disease Y/N : ").lower()
    if f6 == 'y':
        f6=1
    else:
        f6=0

    f7 = input("Do you have fatigue Y/N : ").lower()
    if f7 == 'y':
        f7=1
    else:
        f7=0

    f8 = input("Do you have allergy Y/N : ").lower()
    if f8 == 'y':
        f8=1
    else:
        f8=0

    f9 = input("Do have wheezing Y/N : ").lower()
    if f9 == 'y':
        f9=1
    else:
        f9=0

    f10 = input("Do you consume alcahol Y/N : ").lower()
    if f10 == 'y':
        f10=1
    else:
        f10=0

    f11 = input("Do you have coughing Y/N : ").lower()
    if f11 == 'y':
        f11=1
    else:
        f11=0

    f12 = input("Do you have shortness of breath Y/N : ").lower()
    if f12 == 'y':
        f12=1
    else:
        f12=0

    f13 = input("Do you have swallowing difficulty Y/N : ").lower()
    if f13 == 'y':
        f13=1
    else:
        f13=0

    f14 = input("At last , do you have cheast pain Y/N : ").lower()
    if f14 == 'y':
        f14=1
    else:
        f14=0

    DSet = np.array([[f1, age ,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14]] , dtype=float)
    DSet = scaler.fit_transform(DSet)

    return DSet

print("==================================== Evaluating the model ================================================")

edset = take_data()
pred = model.predict(edset)

classes = {0:"No , you don't have lung cancer." , 1:"Yes , you have lung cancer."}

print(f'''\n
features = {edset}
prediction : {classes.get(int(pred[0]))}

Disclaimer : Not every prediction is accurate . For serious issue please consult a real doctor this model is only educational purposes.
''')
