import streamlit as st
import random
import pandas as pd
#from textblob import TextBlob

# Set up the Streamlit page configuration (call this first)
st.set_page_config(page_title="Customer Review", layout="centered")

# Title and description
st.title("Customer Review Analyzer")
st.write("Analyze customer review. Input text below to see whether the review is positive, negative, or neutral.")

# Input text area
st.subheader("Input Text")
input_text = st.text_area("Enter text for review analysis")

# Analyze sentiment when the button is clicked
if st.button("Analyze Review"):
    if input_text.strip():
        # Perform sentiment analysis
        blob = TextBlob(input_text)
        #sentiment = blob.sentiment

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
