import streamlit as st
import pickle
import pandas as pd 
import numpy as np 
from tensorflow.keras.models import  load_model # type: ignore


st.markdown(
"""
<style>
.st-emotion-cache-sf6nut  {
   color:rgb(32, 69, 112);
    
}
.st-emotion-cache-8g2p72 {
    color: cyan;
    font-weight: bolder;
}
</style>

""",
unsafe_allow_html=True
)
#load the modle 

model=load_model('model.h5')

with open("gen_encoder.pkl",'rb') as file:
    gen_encoder=pickle.load(file)

with open("geo_encoder.pkl",'rb') as file:
    geo_encoder=pickle.load(file)

with open("scalar.pkl",'rb') as file:
    scalar=pickle.load(file)


st.title("Customer chur prediction",text_alignment='center',width='stretch')


st.markdown("""
<style>
div.stslider > CreditScore {
    background-color: #4CAF50;
    color: white;
}
</style>
            
""", unsafe_allow_html=True)

CreditScore=st.slider(label="Credit_Score",min_value=0,max_value=850)
st.write("selected value:",CreditScore)

Geography=s2=st.selectbox(label="City",options=["France","Germany","Spain"])


Gender=st.selectbox(label="Gender",options=['Male','Female'])
Age=st.number_input(label='Age',placeholder="Enter you current age")
if Age<0:
    st.write("Age cannot be negative")
    
Tenure=st.slider(label="Tenure",min_value=1,max_value=10)
st.write("selected value:",Tenure)     


Balance=st.number_input(label="Balance",min_value=0.00)


NumOfProducts=st.slider(label="NumOfProducts",min_value=0,max_value=10)
HasCrCard=st.selectbox(label="HasCrCard",options=[0,1])
IsActiveMember=st.selectbox(label="IsActiveMember",options=[0,1])


EstimatedSalary=st.number_input(label="EstimatedSalary",min_value=0.00)


data={
     	"CreditScore":CreditScore,
         "Geography":Geography,
        "Gender": Gender,
         "Age":Age,
        "Tenure":Tenure,
       "Balance":Balance,
        "NumOfProducts":NumOfProducts,
         "HasCrCard":HasCrCard,	
         "IsActiveMember":IsActiveMember,
         "EstimatedSalary":EstimatedSalary
}

data_df=pd.DataFrame([data])

df_geo=geo_encoder.transform(pd.DataFrame([[data['Geography']]],columns=['Geography'])).toarray()
l1=pd.DataFrame(df_geo,columns=geo_encoder.get_feature_names_out(['Geography']))
geo_df=pd.DataFrame(l1)

data_df.drop('Geography',axis=1,inplace=True)
data_df['Gender']=gen_encoder.transform(data_df['Gender'])
data_df=pd.concat([data_df,geo_df],axis=1)

data_df_scaled=scalar.transform(data_df)

output=model.predict(data_df_scaled)[0][0]

if output>0.5:
    st.write("The customer is  likly to churn")
else:
     st.write("The customer is not  likly to churn")
















