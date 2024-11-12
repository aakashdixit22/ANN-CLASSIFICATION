## integrating ANN model with streamlit app
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder,OneHotEncoder
import pandas as pd
import numpy as np
import pickle

# Load the  trained model
model = tf.keras.models.load_model('model.h5')
with open('label_encoder.pkl','rb') as file:
    label_encoder=pickle.load(file)#load the label encoder  
with open('onehot_encoder.pkl','rb') as file:
    onehot_encoder=pickle.load(file)#load the onehot encoder
with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)#load the scaler

##streamlit app
st.title('ANN Classification(Customer churn predicition)')
# User input 
geography = st.selectbox('Geography',onehot_encoder.categories_[0])
gender= st.selectbox('Gender',label_encoder.classes_)
age = st.slider('Age',18,100,)
tenure = st.slider('Tenure',0,10)
balance = st.number_input('Balance')
num_of_products = st.slider('Number of Products',1,4)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is Active Member',[0,1])
estimated_salary = st.number_input('Estimated Salary')
credit_score = st.slider('Credit Score')

#input data
input_data=pd.DataFrame({
    'CreditScore':[credit_score],
    
    'Gender':[label_encoder.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary]
})

#one hot encode the 'Geography'
geo_encoded=onehot_encoder.transform(np.array(geography).reshape(-1,1)).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded,columns=onehot_encoder.get_feature_names_out(['Geography']))
#combine input data and geography
input_data=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)
input_data_scaled=scaler.transform(input_data)#scale the input data

#predict the churn
prediction=model.predict(scaler.transform(input_data_scaled))
prediction_probab=prediction[0][0]
st.write('Prediction Probability:',prediction_probab)

if prediction_probab>0.5:
    st.write('Customer is likely to churn')
else:
    st.write('Customer is not likely to churn')

