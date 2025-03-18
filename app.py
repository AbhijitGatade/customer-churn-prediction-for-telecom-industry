import streamlit as st
import pandas as pd
import pickle as pkl

st.title("Customer Churn Prediction")
ds = pd.read_csv("cleaned_data.csv")
label_encoders = pkl.load(open("Encoder.pkl", "rb"))
algo = pkl.load(open("SVC.pkl", "rb"))

genders = ds["gender"].unique()
SeniorCitizens = ds["SeniorCitizen"].unique()

mydata = {     
        "gender":"Male",
        "SeniorCitizen":0,
        "Partner":"No",
        "Dependents":"No",
        "tenure":34,
        "PhoneService":"Yes",
        "MultipleLines":"No",
        "InternetService":"DSL",
        "OnlineSecurity":"No",
        "OnlineBackup":"Yes", 
        "DeviceProtection":"No", 
        "TechSupport":"No",
        "StreamingMovies":"No",
        "Contract":"Month-to-month", 
        "PaperlessBilling":"Yes"
}
for column in mydata:
    mydata[column] = st.selectbox("Select " + column + ":", sorted(ds[column].unique()))

if st.button("Predict Customer Churn"):
    data = []
    for column in mydata:
        if column in label_encoders:
            data.append(label_encoders[column].transform([mydata[column]])[0])
        else:
            data.append(mydata[column])
    myinput = pd.DataFrame(data = [data], columns = mydata)
    if algo.predict(myinput)[0] == 0:
        st.write("No, customer will not churn")
    else:
        st.write("Yes, customer will churn")
