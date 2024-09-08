import numpy as np
import streamlit as st
import streamlit.components.v1 as stc
import pickle

with open('Random Forest_model.pkl', 'rb') as file:
    Multi_Class_Classification_Model = pickle.load(file)

html_temp = """ <div style="background-color:#000;padding:10px;border-radius:10px">
                <h1 style="color:#fff;text-align:center">Custromer Segmentation Prediction App</h1>
                <h4 style="color:#fff;text-align:center">Made for: Sales Team</h4>
                """

desc_temp = """ #### Customer Segmentation Prediction App
                This app is used by sales team for deciding Customer Segmentation
                
                Data Source
                Kaggle: Link <https://www.kaggle.com/datasets/kaushiksuresh147/customer-segmentation/data>
                """

profession_mapping = [
    'profession_Artist', 'profession_Doctor', 'profession_Engineer',
    'profession_Entertainment', 'profession_Executive', 'profession_Healthcare',
    'profession_Homemaker', 'profession_Marketing', 'profession_Lawyer']

spending_mapping = {
    'Low' : 0,
    'Average' : 1,
    'High' : 2
}

var_1_mapping = [
    'var_1_Cat_1', 'var_1_Cat_2', 'var_1_Cat_3',
    'var_1_Cat_4', 'var_1_Cat_5', 'var_1_Cat_6', 'var_1_Cat_7']

def main():
    stc.html(html_temp)
    menu = ["Home", "Machine Learning App"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
        st.markdown(desc_temp, unsafe_allow_html=True)
    elif choice == "Machine Learning App":
        run_ml_app()

def run_ml_app():
    design = """<div style="padding:15px;">
                    <h1 style="#fff;>Customer Segmentation</h1>
                </div>
            """
    st.markdown(design, unsafe_allow_html=True)
    left, right = st.columns((2,2))
    gender = left.selectbox('Gender', ('Male', 'Female'))
    married = right.selectbox('Married', ('Yes', 'No'))
    age = left.number_input('Customer Age', min_value= 0, max_value= 150, step= 1)
    education = right.selectbox('Education', ('Graduate', 'Non-Graduate'))
    profession = left.selectbox('Customer Profession', list(profession_mapping))
    work_experience = right.number_input('Customer Work Experience', min_value= 0, step= 1)
    spending = left.selectbox('Spending', list(spending_mapping.keys()))
    family_size = right.number_input('Family Size', min_value= 0, step= 1)
    var_1  = st.selectbox('var_1', list(var_1_mapping))
    button = st.button("Predict")

    #If button is clicked
    if button:
        result = predict(gender, married, age, education, profession, work_experience, spending, family_size, var_1)

        st.success(f'Customer classified as {result} category')

def predict(gender, married, age, education, profession, work_experience, spending, family_size, var_1):
    # Process user input
    gen = 1 if gender == 'Male' else 0
    mar = 1 if married == 'Yes' else 0
    edu = 1 if education == "Graduate" else 0
    spend = spending_mapping[spending]
    profession_encoded = [1 if f == profession else 0 for f in profession_mapping]
    var_1_encoded = [1 if v == var_1 else 0 for v in var_1_mapping]

    features = [gen, mar, age, edu, work_experience, spend, family_size] + profession_encoded + var_1_encoded

    features_array = np.array(features).reshape(1, -1)
    # Making prediction
    prediction = Multi_Class_Classification_Model.predict(features_array)

    # Map the prediction to class labels
    label_mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    prediction_label = label_mapping[prediction[0]]
    
    return prediction_label


if __name__ == "__main__":
    main()