import streamlit as st
import requests
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
from langdetect import detect
import re
from textblob import TextBlob

# Set up the Google API keys and Custom Search Engine ID
API_KEY = st.secrets["GOOGLE_API_KEY"]  # Your Google API key from Streamlit secrets
CX = st.secrets["GOOGLE_SEARCH_ENGINE_ID"]  # Your Google Custom Search Engine ID

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

    # Return the preprocessed text
    return " ".join(filtered_tokens)

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

            # Perform the search query
            response = service.cse().list(q=processed_content, cx=CX, num=10).execute()  # 10 results per page

            # Extract URLs from the search results
            search_results = response.get('items', [])
            detected_matches = []

            # Loop through the search results (including pagination)
            while response.get("queries", {}).get("nextPage"):
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

                        # If similarity exceeds a threshold, record the match
                        if similarity[0][0] > 0.7:  # Adjust threshold for better recall
                            detected_matches.append((url, similarity[0][0], web_text[:500]))  # Display snippet

            # Display results
            if detected_matches:
                st.success("Potential copyright violations detected!")
                for match in detected_matches:
                    st.write(f"- **URL**: {match[0]} - **Similarity**: {match[1]:.2f}")
                    st.write(f"Snippet: {match[2]}...")

                    # Sentiment analysis of the matched content (optional feature)
                    blob = TextBlob(match[2])
                    sentiment = blob.sentiment.polarity
                    sentiment_status = "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"
                    st.write(f"Sentiment of the matched content: {sentiment_status}")

            else:
                st.info("No matches found.")

        except Exception as e:
            st.error(f"Error: {e}")
