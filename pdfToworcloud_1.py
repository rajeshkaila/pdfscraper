import streamlit as st
from PyPDF2 import PdfReader
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Ensure NLTK data is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def preprocess_text(text):
    """Preprocess text by tokenizing and removing stop words."""
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())  # Tokenize and convert to lowercase
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
    return " ".join(filtered_words)

def generate_wordcloud(text):
    """Generate a word cloud from text."""
    wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=500).generate(text)
    return wordcloud

def main():
    st.title("PDF to WordCloud")

    st.write("Upload a PDF file, and this app will generate a word cloud from its text content.")

    pdf_file = st.file_uploader("Upload PDF", type=["pdf"])

    if pdf_file is not None:
        # Extract text from PDF
        with st.spinner("Extracting text from PDF..."):
            text = extract_text_from_pdf(pdf_file)

        if text.strip():
            # Preprocess text
            with st.spinner("Preprocessing text..."):
                preprocessed_text = preprocess_text(text)

            # Generate word cloud
            with st.spinner("Generating WordCloud..."):
                wordcloud = generate_wordcloud(preprocessed_text)

            # Display word cloud
            st.write("## WordCloud")
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)

        else:
            st.error("The uploaded PDF does not contain any readable text.")

if __name__ == "__main__":
    main()
