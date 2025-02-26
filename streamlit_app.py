import streamlit as st
import requests
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from langdetect import detect
import re
from textblob import TextBlob
from wordcloud import WordCloud
from textstat import textstat
import time
import pandas as pd
import matplotlib.pyplot as plt
import warnings

# Suppress XML parsing warnings
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# Set up the Google API keys and Custom Search Engine ID
API_KEY = st.secrets["GOOGLE_API_KEY"]  # Your Google API key from Streamlit secrets
CX = st.secrets["GOOGLE_SEARCH_ENGINE_ID"]  # Your Google Custom Search Engine ID

# Initializing session state for detected matches
if 'detected_matches' not in st.session_state:
    st.session_state.detected_matches = []

# Custom CSS to style the page
hide_streamlit_style = """
    <style>
        .css-1r6p8d1 {display: none;} 
        .css-1v3t3fg {display: none;} 
        header {visibility: hidden;} 
        .css-1tqja98 {visibility: hidden;} 
        .stTextInput>div>div>input {background-color: #f0f0f5; border-radius: 10px;}
        .stButton>button {background-color: #5e35b1; color: white; border-radius: 10px; padding: 10px 20px;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Streamlit UI for title and description
st.title("🔎 Advanced Copyright Content Detection Tool")
st.markdown("""
Detect if your copyrighted content is being used elsewhere on the web.
We use sophisticated algorithms to analyze web content for similarities. 
Simply paste your content and let us do the rest!
""")

# Add input text area
user_content = st.text_area("Paste your copyrighted content:", height=200, placeholder="Enter your text here...")

# Self-hosting and Source Code link
    st.markdown(
        """
        ### Self-Hosting
        If you want to self-host this application or download the source code, please visit:  
            👉 [Download Source Code](https://dhruvbansal8.gumroad.com/l/hhwbm?_gl=1*1hk16wi*_ga*MTQwNDE3ODM4My4xNzM0MzcyNTUw*_ga_6LJN6D94N6*MTc0MDU4NTEwMi4yNS4xLjE3NDA1ODY3NDMuMC4wLjA.)
        """)

# Detect language of the user input
if user_content:
    lang = detect(user_content)
    st.write(f"Detected language: {lang}")

# Pre-process text to improve matching
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    stop_words = set([
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves',
        'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
        'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
        'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
        'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up',
        'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
        'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
        'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
    ])
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(filtered_tokens)

# Add the button for searching copyright violations
if st.button("🔍 Search the Web for Copyright Violations"):
    if not user_content.strip():
        st.error("Please provide your copyrighted content.")
    else:
        with st.spinner('⏳ Searching for potential copyright violations...'):
            try:
                service = build("customsearch", "v1", developerKey=API_KEY)
                processed_content = preprocess_text(user_content)
                response = service.cse().list(q=processed_content, cx=CX, num=10).execute()

                # Reset session state for detected matches
                st.session_state.detected_matches = []

                for result in response.get('items', []):
                    url = result['link']
                    st.write(f"📄 Analyzing {url}...")

                    # Fetch the content from the URL
                    content_response = requests.get(url, timeout=10)
                    if content_response.status_code == 200:
                        web_content = content_response.text
                        soup = BeautifulSoup(web_content, "html.parser")
                        paragraphs = soup.find_all("p")
                        web_text = " ".join([para.get_text() for para in paragraphs])

                        if len(web_text.split()) > 50:  # Ensure the page contains enough text to analyze
                            # Preprocess web content
                            processed_web_text = preprocess_text(web_text)

                            # Calculate similarity between user content and web content
                            vectorizer = TfidfVectorizer().fit_transform([processed_content, processed_web_text])
                            similarity = cosine_similarity(vectorizer[0:1], vectorizer[1:2])

                            # Adjust threshold for better recall (lower the threshold for better detection)
                            if similarity[0][0] > 0.3:  # Changed from 0.5 to 0.3 for higher recall
                                st.session_state.detected_matches.append((url, similarity[0][0], web_text[:500]))

                # Display the results in a professional dashboard layout
                if st.session_state.detected_matches:
                    st.success("🚨 Potential copyright violations detected!")
                    dashboard_columns = st.columns([1, 1, 2])

                    # Display a summary of detected matches
                    with dashboard_columns[0]:
                        st.subheader("📊 Detected Matches Summary")
                        total_matches = len(st.session_state.detected_matches)
                        st.write(f"**Total matches found**: {total_matches}")
                        st.write(f"**Displaying top {min(total_matches, 10)} matches**")

                    # Display snippet samples
                    with dashboard_columns[1]:
                        st.subheader("📝 Snippet Samples")
                        for match in st.session_state.detected_matches[:5]:
                            st.write(f"**URL**: {match[0]} - **Similarity**: {match[1]:.2f}")
                            st.write(f"**Snippet**: {match[2]}...")

                    # Word cloud visualization
                    with dashboard_columns[2]:
                        st.subheader("🌐 Word Cloud of Matches")
                        text = " ".join([match[2] for match in st.session_state.detected_matches])
                        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
                        plt.figure(figsize=(8, 8))
                        plt.imshow(wordcloud, interpolation='bilinear')
                        plt.axis('off')
                        st.pyplot(plt)

                    # Provide a download link for CSV file
                    def convert_df(df):
                        return df.to_csv(index=False).encode('utf-8')

                    df = pd.DataFrame(st.session_state.detected_matches, columns=["URL", "Similarity", "Snippet"])
                    csv = convert_df(df)

                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="detected_matches.csv",
                        mime="text/csv"
                    )

                else:
                    st.info("ℹ️ No matches found.")

            except Exception as e:
                st.error(f"❌ Error: {e}")
