import streamlit as st
from transformers import pipeline

model_dir = '/Users/aryamannsrivastava/Desktop/IMPORTANT/Sentiment Analysis/Assignment-2/modelA2'  # Path to my model
classifier = pipeline('text-classification', model=model_dir)

st.title("Sentiment Analysis using BERT")
text_input = st.text_area("Enter text for sentiment analysis:", "I love Streamlit!")

if st.button('Predict Sentiment'):
    if text_input:
        result = classifier(text_input)  # Getting the model result
        label = result[0]['label']
        score = result[0]['score']
        
        # Display result as "Positive" if label is 'LABEL_1', "Negative" if label is 'LABEL_0'
        if label == 'LABEL_1':
            sentiment = "Positive"
        elif label == 'LABEL_0':
            sentiment = "Negative"
        else:
            sentiment = "Unknown"  # in case the model produces an unknown label
        
        st.write(f"Prediction: {sentiment}")
        st.write(f"Confidence Score: {score:.4f}")
