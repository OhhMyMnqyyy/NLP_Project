import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Function to load and process the dataset
def load_data(file):
    df = pd.read_csv(file)
    return df

# Streamlit app layout
def main():
    st.title("Sentiment Analysis from Dataset")
    st.subheader("Upload your dataset to analyze sentiment trends")

    # File uploader
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        # Load the dataset
        data = load_data(uploaded_file)
        st.write("Dataset preview:")
        st.write(data.head())

        # Check if required columns exist
        if "Tweet" in data.columns and "Sentiment" in data.columns:
            # Display sentiment distribution
            sentiment_counts = data["Sentiment"].value_counts()
            st.write("Sentiment distribution:")
            st.bar_chart(sentiment_counts)

            # Filter by sentiment (optional feature)
            sentiment_filter = st.selectbox(
                "Filter by sentiment:",
                options=["All"] + list(sentiment_counts.index),
                index=0,
            )

            if sentiment_filter != "All":
                filtered_data = data[data["Sentiment"] == sentiment_filter]
                st.write(f"Filtered tweets ({sentiment_filter}):")
                st.write(filtered_data)
            else:
                st.write("All tweets displayed.")

        else:
            st.error("The dataset must contain 'Tweet' and 'Sentiment' columns.")

if __name__ == "__main__":
    main()

streamlit run app.py
