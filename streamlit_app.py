import streamlit as st
import requests
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
from io import StringIO
from langdetect import detect
from pdfminer.high_level import extract_text
from PIL import Image

# Set up the Google API keys and Custom Search Engine ID
API_KEY = st.secrets["GOOGLE_API_KEY"]  # Your Google API key from Streamlit secrets
CX = st.secrets["GOOGLE_SEARCH_ENGINE_ID"]  # Your Google Custom Search Engine ID
TINEYE_API_KEY = st.secrets["TINEYE_API_KEY"]  # Your TinEye API key from Streamlit secrets

# Streamlit UI for text input and file upload
st.title("Advanced Copyright Content Detection Tool")
st.write("Detect if your copyrighted content is being used elsewhere on the web.")

# Option for user to choose content type: text, image, or file
content_type = st.selectbox("Select Content Type", ["Text", "Image", "File"])

# Handle Text Input
if content_type == "Text":
    user_content = st.text_area("Paste your copyrighted content:", height=200)

    # Language detection for multilingual content
    if user_content:
        lang = detect(user_content)
        st.write(f"Detected language: {lang}")

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

                        # Clean and parse the HTML content
                        soup = BeautifulSoup(web_content, "html.parser")
                        paragraphs = soup.find_all("p")
                        web_text = " ".join([para.get_text() for para in paragraphs])

                        # Calculate similarity between user content and web content
                        vectorizer = TfidfVectorizer().fit_transform([user_content, web_text])
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

# Handle Image Upload and Reverse Image Search (via TinEye)
elif content_type == "Image":
    uploaded_image = st.file_uploader("Upload an image to analyze:", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Perform reverse image search with TinEye
        if st.button("Search the Web for Image Violations"):
            try:
                # Convert the uploaded image to a URL or use a direct TinEye API request
                image_url = upload_image_to_server(uploaded_image)  # Function to handle the upload
                if image_url:
                    search_results = search_tineye(image_url)
                    if search_results:
                        st.success("Potential image copyright violations detected!")
                        for result in search_results.get("results", []):
                            st.write(f"- **URL**: {result['url']} - **Match Confidence**: {result['match']}")
                    else:
                        st.info("No matches found.")
                else:
                    st.error("Failed to upload the image for reverse search.")

            except Exception as e:
                st.error(f"Error: {e}")

# Handle File Upload (PDF, DOCX, etc.)
elif content_type == "File":
    uploaded_file = st.file_uploader("Upload a text document to analyze:", type=["txt", "pdf", "docx"])

    if uploaded_file is not None:
        file_content = ""
        if uploaded_file.type == "text/plain":
            file_content = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
        elif uploaded_file.type == "application/pdf":
            file_content = extract_text(uploaded_file)
        # Add support for DOCX here if needed

        st.write("Content from uploaded file:")
        st.write(file_content)

        # Option to perform search after uploading
        if st.button("Search the Web for Copyright Violations"):
            if not file_content.strip():
                st.error("The uploaded file has no content.")
            else:
                try:
                    # Proceed with the same web search logic as above for files
                    service = build("customsearch", "v1", developerKey=API_KEY)
                    response = service.cse().list(q=file_content, cx=CX).execute()

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

                            # Clean and parse the HTML content
                            soup = BeautifulSoup(web_content, "html.parser")
                            paragraphs = soup.find_all("p")
                            web_text = " ".join([para.get_text() for para in paragraphs])

                            # Calculate similarity between user content and web content
                            vectorizer = TfidfVectorizer().fit_transform([file_content, web_text])
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

# TinEye Reverse Image Search Function
def search_tineye(image_url):
    url = 'https://api.tineye.com/rest/search/'
    params = {
        'image_url': image_url,
        'api_key': TINEYE_API_KEY
    }
    response = requests.get(url, params=params)
    return response.json()

# Function to upload image to a server (use a real file hosting service in production)
def upload_image_to_server(uploaded_image):
    # Temporary method to simulate image URL generation (you should host the image somewhere)
    image_url = "https://example.com/image.jpg"  # Replace with real logic to upload and get a URL
    return image_url
