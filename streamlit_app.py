import streamlit as st
import requests
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set up the Google API keys and Custom Search Engine ID
API_KEY = st.secrets["GOOGLE_API_KEY"]  # Your Google API key from Streamlit secrets
CX = st.secrets["GOOGLE_SEARCH_ENGINE_ID"]  # Your Google Custom Search Engine ID

# Streamlit UI for text input and file upload
st.title("Advanced Copyright Content Detection Tool")
st.write("Detect if your copyrighted content is being used elsewhere on the web.")

# Option for user to choose content type: text or image
content_type = st.selectbox("Select Content Type", ["Text"])

# If user selects "Text"
if content_type == "Text":
    # Text input for copyrighted content
    user_content = st.text_area("Paste your copyrighted content:", height=200)

    if st.button("Search the Web for Copyright Violations"):
        if not user_content.strip():
            st.error("Please provide your copyrighted content.")
        else:
            try:
                # Initialize Google Custom Search API
                service = build("customsearch", "v1", developerKey=API_KEY)

                # Perform the search query
                response = service.cse().list(q=user_content, cx=CX).execute()

                # Extract URLs from the search results
                search_results = response.get('items', [])
                detected_matches = []

                for result in search_results:
                    url = result['link']
                    st.write(f"Analyzing {url}...")

                    # Fetch the content from the URL
                    content_response = requests.get(url, timeout=10)
                    if content_response.status_code == 200:
                        web_content = content_response.text

                        # Calculate similarity between user content and web content
                        vectorizer = TfidfVectorizer().fit_transform([user_content, web_content])
                        similarity = cosine_similarity(vectorizer[0:1], vectorizer[1:2])

                        # If similarity exceeds a threshold, record the match
                        if similarity[0][0] > 0.8:  # Adjust the threshold as needed
                            detected_matches.append((url, similarity[0][0]))

                # Display results
                if detected_matches:
                    st.success("Potential copyright violations detected!")
                    for match in detected_matches:
                        st.write(f"- **URL**: {match[0]} - **Similarity**: {match[1]:.2f}")
                else:
                    st.info("No matches found.")

            except Exception as e:
                st.error(f"Error: {e}")

# If no content is selected or uploaded, show a message
else:
    st.info("Please select content type and provide the necessary input (text).")
