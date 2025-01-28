import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Streamlit app title
st.title("Sentiment Analysis for Customer Reviews")
st.subheader("Analyze customer reviews and classify them as Positive, Negative, or Neutral.")

# Step 1: File Upload
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file:
    try:
        # Step 2: Load Dataset
        data = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.dataframe(data.head())

        # Ensure the required columns are present
        if 'Review' in data.columns and 'Sentiment' in data.columns:
            st.success("Dataset is valid and ready to process.")

            # Step 3: Preprocessing
            X = data['Review']
            y = data['Sentiment']

            # Split data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Vectorize text data
            vectorizer = CountVectorizer()
            X_train_vec = vectorizer.fit_transform(X_train)
            X_test_vec = vectorizer.transform(X_test)

            # Step 4: Train Naive Bayes Model
            model = MultinomialNB()
            model.fit(X_train_vec, y_train)

            # Step 5: Model Evaluation
            y_pred = model.predict(X_test_vec)
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Model Accuracy: **{accuracy * 100:.2f}%**")

            # Step 6: Test the Model
            st.subheader("Test the Model with Your Input")
            user_input = st.text_area("Enter a customer review to analyze:")

            if st.button("Analyze Sentiment"):
                if user_input.strip():
                    input_vec = vectorizer.transform([user_input])
                    prediction = model.predict(input_vec)[0]
                    st.write(f"Predicted Sentiment: **{prediction.capitalize()}**")
                else:
                    st.warning("Please enter a review to analyze.")
        else:
            st.error("The dataset must contain 'Review' and 'Sentiment' columns.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload a CSV file to proceed.")

# Step 7: Footer Information
st.markdown("""
---
**Notes:**
1. The uploaded CSV must contain two columns: `Review` (text of the review) and `Sentiment` (labels like `positive`, `negative`, or `neutral`).
2. Example reviews:
    - Positive: "This product is amazing!"
    - Negative: "Terrible experience, never buying again."
    - Neutral: "It was okay, not great but not bad."
""")
