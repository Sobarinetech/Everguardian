import streamlit as st
import requests
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
from langdetect import detect
import re
from textblob import TextBlob
from wordcloud import WordCloud
from textstat import textstat
import time
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from gensim.summarization import summarize
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from difflib import SequenceMatcher

# Download required NLTK data
nltk.download('vader_lexicon')
nltk.download('punkt')

# Set up the Google API keys and Custom Search Engine ID
API_KEY = st.secrets["GOOGLE_API_KEY"]  # Your Google API key from Streamlit secrets
CX = st.secrets["GOOGLE_SEARCH_ENGINE_ID"]  # Your Google Custom Search Engine ID

# Initializing session state for detected matches
if 'detected_matches' not in st.session_state:
    st.session_state.detected_matches = []

# Streamlit UI for text input
st.title("Advanced Copyright Content Detection Tool")
st.write("Detect if your copyrighted content is being used elsewhere on the web.")

# Option for user to input text
user_content = st.text_area("Paste your copyrighted content:", height=200)

# Language detection for multilingual content
if user_content:
    lang = detect(user_content)
    st.write(f"Detected language: {lang}")

# Pre-process text to improve matching
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenize the text by splitting into words
    tokens = text.split()

    # Remove common stopwords manually (instead of using NLTK)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Return the preprocessed text
    return " ".join(filtered_tokens)

# Function to calculate similarity using SequenceMatcher
def calculate_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

# Function to display key phrases
def display_key_phrases(text):
    tokens = nltk.word_tokenize(text)
    word_freq = Counter(tokens)
    common_words = word_freq.most_common(10)
    st.write("Key Phrases:")
    for word, freq in common_words:
        st.write(f"{word}: {freq} times")

# Function to summarize text
def summarize_text(text):
    try:
        summary = summarize(text)
        return summary
    except ValueError:
        return text

# Button to search for copyright violations
if st.button("Search the Web for Copyright Violations"):
    if not user_content.strip():
        st.error("Please provide your copyrighted content.")
    else:
        try:
            # Initialize Google Custom Search API
            service = build("customsearch", "v1", developerKey=API_KEY)

            # Preprocess user content before searching
            processed_content = preprocess_text(user_content)

            # Perform the search query with num=10 to fetch the first 10 results
            response = service.cse().list(q=processed_content, cx=CX, num=10).execute()  # Fetch first 10 results

            # Reset detected matches
            st.session_state.detected_matches = []

            # Extract URLs from the first page of search results
            for result in response.get('items', []):
                url = result['link']
                st.write(f"Analyzing {url}...")

                # Fetch the content from the URL
                content_response = requests.get(url, timeout=10)
                if content_response.status_code == 200:
                    web_content = content_response.text

                    # Clean and parse the HTML content
                    soup = BeautifulSoup(web_content, "html.parser")
                    paragraphs = soup.find_all("p")
                    web_text = " ".join([para.get_text() for para in paragraphs])

                    # Preprocess web content
                    processed_web_text = preprocess_text(web_text)

                    # Calculate similarity between user content and web content
                    vectorizer = TfidfVectorizer().fit_transform([processed_content, processed_web_text])
                    similarity = cosine_similarity(vectorizer[0:1], vectorizer[1:2])

                    # Additional similarity measure
                    seq_similarity = calculate_similarity(processed_content, processed_web_text)

                    # If similarity exceeds a threshold, record the match
                    if similarity[0][0] > 0.7 or seq_similarity > 0.7:  # Adjust threshold for better recall
                        st.session_state.detected_matches.append((url, similarity[0][0], seq_similarity, web_text[:500]))  # Display snippet

            # Fetch the next 15 results via pagination (if necessary)
            if len(st.session_state.detected_matches) < 25 and response.get("queries", {}).get("nextPage"):
                next_page_token = response["queries"]["nextPage"][0]["startIndex"]
                response = service.cse().list(q=processed_content, cx=CX, num=10, start=next_page_token).execute()

                for result in response.get('items', []):
                    url = result['link']
                    st.write(f"Analyzing {url}...")

                    # Fetch the content from the URL
                    content_response = requests.get(url, timeout=10)
                    if content_response.status_code == 200:
                        web_content = content_response.text

                        # Clean and parse the HTML content
                        soup = BeautifulSoup(web_content, "html.parser")
                        paragraphs = soup.find_all("p")
                        web_text = " ".join([para.get_text() for para in paragraphs])

                        # Preprocess web content
                        processed_web_text = preprocess_text(web_text)

                        # Calculate similarity between user content and web content
                        vectorizer = TfidfVectorizer().fit_transform([processed_content, processed_web_text])
                        similarity = cosine_similarity(vectorizer[0:1], vectorizer[1:2])

                        # Additional similarity measure
                        seq_similarity = calculate_similarity(processed_content, processed_web_text)

                        # If similarity exceeds a threshold, record the match
                        if similarity[0][0] > 0.7 or seq_similarity > 0.7:  # Adjust threshold for better recall
                            st.session_state.detected_matches.append((url, similarity[0][0], seq_similarity, web_text[:500]))  # Display snippet

                    # Stop if we have reached the limit of 25 results
                    if len(st.session_state.detected_matches) >= 25:
                        break

            # Display results
            if st.session_state.detected_matches:
                st.success("Potential copyright violations detected!")
                for match in st.session_state.detected_matches:
                    st.write(f"- **URL**: {match[0]} - **TF-IDF Similarity**: {match[1]:.2f} - **Sequence Similarity**: {match[2]:.2f}")
                    st.write(f"Snippet: {match[3]}...")

                    # Sentiment analysis of the matched content (optional feature)
                    blob = TextBlob(match[3])
                    sentiment = blob.sentiment.polarity
                    sentiment_status = "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"
                    st.write(f"Sentiment of the matched content: {sentiment_status}")

                    # Display a word cloud of the matched content
                    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(match[3])
                    plt.figure(figsize=(10, 5))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis('off')
                    st.pyplot(plt)

                    # Display key phrases
                    display_key_phrases(match[3])

                    # Readability score of the matched content
                    readability_score = textstat.flesch_reading_ease(match[3])
                    st.write(f"Readability score: {readability_score}")

                    # Summarize the matched content
                    summary = summarize_text(match[3])
                    st.write(f"Summary of the matched content: {summary}")

                    # Delay to avoid rate limiting
                    time.sleep(1)

            else:
                st.info("No matches found.")

        except Exception as e:
            st.error(f"Error: {e}")

# Option to save results to a CSV file
if st.button("Save Results to CSV"):
    if st.session_state.detected_matches:
        df = pd.DataFrame(st.session_state.detected_matches, columns=["URL", "TF-IDF Similarity", "Sequence Similarity", "Snippet"])
        df.to_csv("detected_matches.csv", index=False)
        st.success("Results saved to detected_matches.csv")
    else:
        st.error("No results to save.")
