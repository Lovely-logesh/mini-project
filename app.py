import streamlit as st
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras

# Hardcoded username and password hashes for demonstration
USERS = {
    "Survivor": "5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8",  # Hash of "admin"
    "Student": "2b71b89b75a0e1313a088e7530cb0e1f1edf5f4c52091c78b0f48f0a6b51e768",  # Hash of "admin"
    "User": "3ef80ca059b60870736b4b1ca76a3e59f20ca2e3e13c07fcbcd7f8892f27571d",    # Hash of "admin"
}

# Function to hash passwords using hashlib
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Function to authenticate user
def login(username, password):
    hashed_password = hash_password(password)
    if username in USERS and hashed_password == USERS[username]:
        session_state.logged_in = True
        session_state.username = username
        st.success(f"Logged In as {username}")
    else:
        st.error("Incorrect username or password")

# Function to logout user
def logout():
    session_state.logged_in = False
    session_state.username = ""
    st.success("Logged Out")

# Initialize session state
class SessionState:
    def __init__(self):
        self.logged_in = False
        self.username = ""

# Main function to run the app
def main():
    st.title("Breast Tumour Classification using Deep Learning")

    # Check if user is logged in
    if not session_state.logged_in:
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            login(username, password)
    else:
        st.subheader(f"Welcome, {session_state.username}!")

        # Load breast cancer dataset
        breast_cancer_dataset = load_breast_cancer()
        data_frame = pd.DataFrame(breast_cancer_dataset.data, columns=breast_cancer_dataset.feature_names)
        data_frame.drop(data_frame.columns[-20:], axis=1, inplace=True)
        data_frame['label'] = breast_cancer_dataset.target

        # Display dataset on checkbox click
        if st.checkbox('Show Dataset'):
            st.dataframe(data_frame)

        # Model input form
        st.subheader("Enter Input Values")
        with st.expander("Input Features"):
            mean_radius = st.number_input("Mean Radius", value=20.0)
            mean_texture = st.number_input("Mean Texture", value=16.0)
            mean_perimeter = st.number_input("Mean Perimeter", value=85.0)
            mean_area = st.number_input("Mean Area", value=550.0)
            mean_smoothness = st.number_input("Mean Smoothness", value=0.1)
            mean_compactness = st.number_input("Mean Compactness", value=0.1)
            mean_concavity = st.number_input("Mean Concavity", value=0.1)
            mean_concave_points = st.number_input("Mean Concave Points", value=0.05)
            mean_symmetry = st.number_input("Mean Symmetry", value=0.1)
            mean_fractal_dimension = st.number_input("Mean Fractal Dimension", value=0.02)

        # Display results on button click
        if st.button("Classify"):
            # Preprocessing and model building
            X = data_frame.drop(columns='label', axis=1)
            Y = data_frame['label']
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
            scaler = StandardScaler()
            X_train_std = scaler.fit_transform(X_train)
            X_test_std = scaler.transform(X_test)
            model = keras.Sequential([
                keras.layers.Flatten(input_shape=(X.shape[1],)),
                keras.layers.Dense(20, activation='relu'),
                keras.layers.Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer='adam',
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
            history = model.fit(X_train_std, Y_train, validation_split=0.1, epochs=10, verbose=1)

            # Evaluate model
            loss, accuracy = model.evaluate(X_test_std, Y_test)
            st.write(f"Model Accuracy: {accuracy}")

            # Make prediction
            input_data = np.array([[mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness,
                                    mean_compactness, mean_concavity, mean_concave_points, mean_symmetry,
                                    mean_fractal_dimension]])
            input_data_std = scaler.transform(input_data)
            prediction = model.predict(input_data_std)[0][0]

            # Display prediction
            if prediction >= 0.5:
                st.success("Predicted: Benign")
            else:
                st.error("Predicted: Malignant")

    # Logout button
    if session_state.logged_in:
        if st.button("Logout"):
            logout()

if __name__ == "__main__":
    session_state = SessionState()
    main()
