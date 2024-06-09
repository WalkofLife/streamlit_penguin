import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Penguin Prediction App

This app predicts the **Palmer Penguin** Species!

""")

st.sidebar.header("User Input Features")
st.sidebar.markdown("""
[Example CSV Input File] (https://github.com/dataprofessor/data/blob/master/penguins_example.csv)
""")

# Collects user Input Features into Dataframe
uploaded_file = st.sidebar.file_uploader('Upload your input CSV file', type = ['csv'])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)

else:
    def user_input_features():
        island = st.sidebar.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
        sex = st.sidebar.selectbox('Gender', ('male', 'female'))
        bill_length_mm = st.sidebar.slider('Bill length (mm)', 32.1, 59.4, 43.9)
        flipper_length_mm = st.sidebar.slider('Flipper length (mm)', 172.0, 231.0, 201.0)
        body_mass_g = st.sidebar.slider('Body mass(g)', 2700.0, 6300.0, 4207.0)

        data = {
            'island' : island,
            'bill_length_mm' : bill_length_mm,
            'flipper_length_mm' : flipper_length_mm,
            'body_mass_g' : body_mass_g,
            'sex': sex
        }
        return pd.DataFrame(data, index = [0])
    input_df = user_input_features()

penguins_raw = pd.read_excel('penguins_cleaned.xlsx')
penguins = penguins_raw.drop(columns=['species'])
# df = pd.concat([input_df, penguins], axis = 1)
# Ensure columns align with user input
columns_needed = list(penguins.columns)
input_df = input_df.reindex(columns=columns_needed, fill_value=0)
df = pd.concat([input_df, penguins])

target = 'species'
encode = ['sex', 'island']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix= col)
    df = pd.concat([df, dummy], axis = 1)
    del df[col]
df = df[:1] # Select only scoring row

# Display the user input features
st.subheader("User Input Features")

if uploaded_file is not None:
    st.write(df)
else:
    st.write("Awaiting CSV file to be uploaded. Currently sing input parameter")
    st.write(df)

load_clf = pickle.load(open('penguins_clf.pkl', 'rb'))

# Apply model to make preditions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)

st.subheader('Predictions')
penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
st.write(penguins_species[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)