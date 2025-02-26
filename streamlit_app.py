import streamlit as st
import requests
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from langdetect import detect
import re
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from wordcloud import WordCloud

# Suppress XML parsing warnings
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# Google API credentials
API_KEY = st.secrets["GOOGLE_API_KEY"]
CX = st.secrets["GOOGLE_SEARCH_ENGINE_ID"]

# Initialize detected matches
if 'detected_matches' not in st.session_state:
    st.session_state.detected_matches = []

# Streamlit UI
st.title("🔎 Advanced Copyright Content Detection Tool")
st.markdown("Detect if your copyrighted content is being used elsewhere on the web.")

# User input
user_content = st.text_area("Paste your copyrighted content:", height=200, placeholder="Enter text here...")

# Detect language
if user_content:
    lang = detect(user_content)
    st.write(f"Detected language: {lang}")

# Text preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
    tokens = text.split()
    return " ".join(tokens[:32])  # Limit to first 32 words

# Button to start search
if st.button("🔍 Search for Copyright Violations"):
    if not user_content.strip():
        st.error("Please provide your copyrighted content.")
    else:
        with st.spinner('⏳ Searching for potential copyright violations...'):
            try:
                service = build("customsearch", "v1", developerKey=API_KEY)
                processed_content = preprocess_text(user_content)
                
                response = service.cse().list(q=processed_content, cx=CX, num=10).execute()
                st.session_state.detected_matches = []

                for result in response.get('items', []):
                    url = result['link']
                    st.write(f"📄 Analyzing {url}...")

                    try:
                        content_response = requests.get(url, timeout=10)
                        if content_response.status_code != 200:
                            st.warning(f"⚠️ Skipping {url} (status {content_response.status_code})")
                            continue

                        soup = BeautifulSoup(content_response.text, "html.parser")
                        paragraphs = soup.find_all("p")
                        web_text = " ".join([para.get_text() for para in paragraphs])[:5000]  # Limit text length

                        if not web_text.strip():
                            st.warning(f"⚠️ No readable text found on {url}")
                            continue

                        # Compute similarity
                        vectorizer = TfidfVectorizer().fit_transform([processed_content, web_text])
                        similarity = cosine_similarity(vectorizer[0:1], vectorizer[1:2])[0][0]

                        if similarity > 0.3:  # Lowered threshold
                            st.session_state.detected_matches.append((url, round(similarity, 2), web_text[:500]))

                    except Exception as e:
                        st.warning(f"⚠️ Error processing {url}: {e}")

                # Display results
                if st.session_state.detected_matches:
                    st.success(f"🚨 {len(st.session_state.detected_matches)} possible matches found!")
                    
                    df = pd.DataFrame(st.session_state.detected_matches, columns=["URL", "Similarity", "Snippet"])
                    st.dataframe(df)

                    # Word cloud
                    st.subheader("🌐 Word Cloud of Matches")
                    text = " ".join([match[2] for match in st.session_state.detected_matches])
                    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
                    plt.figure(figsize=(8, 8))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis('off')
                    st.pyplot(plt)

                    # Download CSV
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download Results as CSV", data=csv, file_name="matches.csv", mime="text/csv")

                else:
                    st.info("ℹ️ No matches found.")

            except Exception as e:
                st.error(f"❌ Error: {e}")
