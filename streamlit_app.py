import streamlit as st
import requests
from bs4 import BeautifulSoup
from PIL import Image
import imagehash
import io
import os
from urllib.parse import urljoin

# Streamlit UI setup
st.title("Image Duplicate Detection Tool")
st.write("Upload an image to find duplicates on the web.")

# File uploader for image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

def get_image_hash(image):
    """Generate a hash for the image to compare similarity."""
    return imagehash.average_hash(image)

def compare_images(image1, image2):
    """Compare two images based on their hash values."""
    hash1 = get_image_hash(image1)
    hash2 = get_image_hash(image2)
    return hash1 - hash2  # The lower the value, the more similar the images

def fetch_images_from_url(url):
    """Fetch image URLs from a webpage using BeautifulSoup."""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    image_urls = []
    for img_tag in soup.find_all("img"):
        img_url = img_tag.get("src")
        if img_url:
            # Handle relative URLs
            img_url = urljoin(url, img_url)
            image_urls.append(img_url)
    return image_urls

if uploaded_image:
    # Load the uploaded image
    image = Image.open(uploaded_image)
    uploaded_image_hash = get_image_hash(image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Web scraping to search for images online
    search_query = st.text_input("Enter a search query to look for similar images:", "")
    
    if st.button("Search for Similar Images"):
        if not search_query:
            st.error("Please provide a search query to look for similar images.")
        else:
            try:
                # Perform a search using Google Images or another image source here (example: scraping a webpage)
                search_results = fetch_images_from_url(f"https://www.google.com/search?hl=en&tbm=isch&q={search_query}")
                
                detected_matches = []
                for img_url in search_results:
                    try:
                        img_response = requests.get(img_url, timeout=5)
                        img = Image.open(io.BytesIO(img_response.content))

                        # Compare hashes
                        if compare_images(image, img) < 5:  # Threshold value can be adjusted
                            detected_matches.append(img_url)
                    except Exception as e:
                        # Skip images that can't be loaded
                        continue

                # Display detected matches
                if detected_matches:
                    st.success("Potential duplicate images detected!")
                    for match in detected_matches:
                        st.write(f"- [Found duplicate image]({match})")
                else:
                    st.info("No duplicates found.")
            
            except Exception as e:
                st.error(f"Error: {e}")

else:
    st.info("Please upload an image to start.")
