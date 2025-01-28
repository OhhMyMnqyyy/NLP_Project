import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# App title
st.title("Sentiment Analysis for Customer Reviews")

# Step 1: Upload the CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    # Step 2: Load the dataset
    data = pd.read_csv(uploaded_file)
    st.write("Preview of the dataset:", data.head())

    if 'Review' in data.columns and 'Sentiment' in data.columns:
        # Step 3: Data Preprocessing
        X = data['Review']
        y = data['Sentiment']

        # Splitting the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Vectorizing text data
        vectorizer = CountVectorizer()
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        # Step 4: Train a Naive Bayes model
        model = MultinomialNB()
        model.fit(X_train_vec, y_train)

        # Step 5: Evaluate the model
        y_pred = model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

        # Step 6: Make Predictions
        st.subheader("Test the Model")
        user_input = st.text_area("Enter a customer review:")

        if st.button("Predict"):
            if user_input:
                input_vec = vectorizer.transform([user_input])
                prediction = model.predict(input_vec)[0]
                st.write(f"Predicted Sentiment: **{prediction}**")
            else:
                st.write("Please enter a review to analyze.")
    else:
        st.error("The uploaded file must contain 'Review' and 'Sentiment' columns.")

streamlit run app.py

