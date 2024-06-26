import streamlit as st
import pandas as pd
import joblib

# Load the trained model and other components
model = joblib.load('logistic_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')
encoder_churn = joblib.load('encoder_churn.pkl')

# Define the prediction function
def predict(input_features):
    input_df = pd.DataFrame(input_features)
    
    # Encode categorical features using the saved label encoders
    for column, encoder in label_encoders.items():
        if column in input_df.columns:
            input_df[column] = encoder.transform(input_df[column])
    
    # Scale numerical features using the saved scaler
    numerical_features = ['Monthly Charges', 'tenure', 'Total Charges']
    input_df[numerical_features] = scaler.transform(input_df[numerical_features])
    
    # Make predictions using the loaded model
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    # Decode the prediction back to original class
    prediction_decoded = encoder_churn.inverse_transform(prediction)

    return prediction_decoded, prediction_proba

# Create the web interface
def main():
    st.title('Customer Churn Prediction')

    st.write('Enter the features below to get predictions:')

    # Streamlit input fields
    feature1 = st.selectbox('Gender', ['Male', 'Female'])
    feature2 = st.number_input('Senior Citizen', min_value=0, max_value=1)
    feature3 = st.selectbox('Partner', ['Yes', 'No'])
    feature4 = st.selectbox('Dependents', ['Yes', 'No'])
    feature5 = st.number_input('tenure', min_value=0)
    feature6 = st.selectbox('Phone Service', ['Yes', 'No'])
    feature7 = st.selectbox('Multiple Lines', ['Yes', 'No', 'No phone service'])
    feature8 = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    feature9 = st.selectbox('Online Security', ['Yes', 'No', 'No internet service'])
    feature10 = st.selectbox('Online Backup', ['Yes', 'No', 'No internet service'])
    feature11 = st.selectbox('Device Protection', ['Yes', 'No', 'No internet service'])
    feature12 = st.selectbox('Tech Support', ['Yes', 'No', 'No internet service'])
    feature13 = st.selectbox('Streaming TV', ['Yes', 'No', 'No internet service'])
    feature14 = st.selectbox('Streaming Movies', ['Yes', 'No', 'No internet service'])
    feature15 = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    feature16 = st.selectbox('Paperless Billing', ['Yes', 'No'])
    feature17 = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    feature18 = st.number_input('Monthly Charges', min_value=0.0)
    feature19 = st.number_input('Total Charges', min_value=0.0)

    # Combine input features into a DataFrame
    input_data = {
        'Gender': [feature1],
        'Senior Citizen': [feature2],
        'Partner': [feature3],
        'Dependents': [feature4],
        'tenure': [feature5],
        'Phone Service': [feature6],
        'Multiple Lines': [feature7],
        'Internet Service': [feature8],
        'Online Security': [feature9],
        'Online Backup': [feature10],
        'Device Protection': [feature11],
        'Tech Support': [feature12],
        'Streaming TV': [feature13],
        'Streaming Movies': [feature14],
        'Contract': [feature15],
        'Paperless Billing': [feature16],
        'Payment Method': [feature17],
        'Monthly Charges': [feature18],
        'Total Charges': [feature19]
    }

    if st.button('Predict'):
        prediction, prediction_proba = predict(input_data)
        st.write('Prediction:', prediction[0])

if __name__ == '__main__':
    main()
