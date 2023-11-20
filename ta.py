import streamlit as st 
import openai
from wordcloud import WordCloud
from dotenv import load_dotenv
import os
import re 
import json
import spacy
from spacy import displacy


load_dotenv()

# Set up OpenAI API credentials
openai.api_key = os.getenv("OPENAI_API_KEY")

HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""

#function for generating the word cloud
def generate_wordcloud(text):
    # Create and generate a word cloud image
    wordcloud = WordCloud(width=800, height=800,
    background_color='black', min_font_size=10).generate(text)
    # Save the wordcloud image to disk
    wordcloud.to_file("wordcloud.png")
    # Return the image path
    return "wordcloud.png"

def ner(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    html =displacy.render(doc, style='ent', jupyter=False)
    html = html.replace("\n\n","\n")
    st.write(HTML_WRAPPER.format(html),unsafe_allow_html=True)

#function to extract key findings from text using Da-Vinci003
def extract_key_findings(text):
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt="Please find the key insights from the below text in maximum of 5 bullet points and also the summary in maximum of 3 sentences:\n"+text,
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0)
    return response['choices'][0]['text']

#function to extract the most positive words from the text using Da-Vinci003
def most_positive_words(text):
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt="Please extract the most positive keywords from the below text\n"+text,
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0)
    return response['choices'][0]['text']

# Streamlit Code
st.set_page_config(layout="wide")

st.title("GPT3 Powered Text Analytics App :page_with_curl:")

with st.expander("About this application"):
    st.markdown("This app is built using the [OpenAI GPT3](https://platform.openai.com/), Streamlit, and Spacy.")


input_text = st.text_area("Enter your text to analyze")

if input_text is not None:
    if st.button("Analyze Text"):
        st.markdown("**Input Text**")
        st.info(input_text)
        col1, col2, col3 = st.columns([1,2,1])
        with col1:
            st.markdown("**Key Findings based on your Text**")
            st.success(extract_key_findings(input_text))
        with col2:
            st.markdown("**Output Text**")
            st.image(generate_wordcloud(input_text))
        with col3:
            st.markdown("**Most Positive Words**")
            st.success(most_positive_words(input_text))
        
        st.markdown("**Named Entity Recognition**")
        ner(input_text)
