import csv
import streamlit as st
import random
import pandas as pd
from textblob import TextBlob

# Set up the Streamlit page configuration
st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="centered")

# Title and description
st.title("Sentiment Analysis Dashboard")
st.write("Analyze the sentiment of your text using TextBlob. Input text below to see whether the sentiment is positive, negative, or neutral.")

# Input text area
st.subheader("Input Text")
input_text = st.text_area("Enter your text here:", height=200)

# Analyze sentiment when the button is clicked
if st.button("Analyze Sentiment"):
    if input_text.strip():
        # Perform sentiment analysis
        blob = TextBlob(input_text)
        sentiment = blob.sentiment

        # Display results
        st.subheader("Results")
        st.write(f"**Polarity:** {sentiment.polarity:.2f} (Ranges from -1 to 1)")
        st.write(f"**Subjectivity:** {sentiment.subjectivity:.2f} (Ranges from 0 to 1)")

        # Provide sentiment classification
        if sentiment.polarity > 0:
            st.success("The sentiment is Positive! 😊")
        elif sentiment.polarity < 0:
            st.error("The sentiment is Negative! 😟")
        else:
            st.info("The sentiment is Neutral. 😐")
    else:
        st.warning("Please enter some text to analyze.")


