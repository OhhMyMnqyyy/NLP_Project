import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# GitHub URL of the CSV file
CSV_URL = "https://raw.githubusercontent.com/yourusername/sentiment-analysis/main/data/customer_reviews.csv"

# Title of the app
st.title("Sentiment Analysis for Customer Reviews")

# Load the dataset directly from GitHub
@st.cache_data
def load_data(url):
    return pd.read_csv(url)

# Load data
st.write("### Loading Dataset...")
data = load_data(CSV_URL)

# Display the data preview
st.write("### Dataset Preview:")
st.dataframe(data.head())

# Check for required columns
if "Review" in data.columns and "Sentiment" in data.columns:
    st.success("Dataset contains required columns: 'Review' and 'Sentiment'.")

    # Preprocess data
    st.write("### Preprocessing Data...")
    X = data["Review"]
    y = data["Sentiment"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Vectorize text data
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train model
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    accuracy = model.score(X_test_vec, y_test)
    st.write(f"Model Accuracy: **{accuracy:.2f}**")

    # Text input for prediction
    st.write("### Test Sentiment Analysis")
    user_input = st.text_area("Enter a review:")
    if st.button("Analyze"):
        user_input_vec = vectorizer.transform([user_input])
        prediction = model.predict(user_input_vec)[0]
        st.write(f"Predicted Sentiment: **{prediction}**")
else:
    st.error("Dataset must contain 'Review' and 'Sentiment' columns.")
