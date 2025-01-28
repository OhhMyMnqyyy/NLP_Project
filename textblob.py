from textblob import TextBlob

text = "Streamlit is an amazing tool!"
blob = TextBlob(text)
print(blob.sentiment)
