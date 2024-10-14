import streamlit as st
from nltk.corpus import stopwords
import PyPDF2
from PyPDF2.errors import PdfReadError
import re
from collections import Counter
from math import log
import nltk.corpus as Corpus
from sklearn.feature_extraction.text import TfidfVectorizer
from pymongo import MongoClient
import os

# MongoDB connection setup
client = MongoClient("mongodb://localhost:27017/")
db = client["pdf_database"]
collection = db["pdf_summary"]

pdf_file_path = st.text_input("Enter the path of PDF : ").replace('"', '').replace("'", '')

def read_pdf_file(file_path):
    try:
        # Attempt to open the file in binary mode
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)

            # Check for an empty or invalid PDF
            if not reader.pages:
                raise ValueError("The PDF file is empty or corrupted.")
                
            # Extracting text from the pdf
            page = len(reader.pages)
            pdftext = []
            for i in range(page):
                pdftext.append(reader.pages[i].extract_text())
            return pdftext, page

    except FileNotFoundError:
        print("The specified PDF file was not found. Please check the file path.")
    except PdfReadError:
        print("The PDF file could not be read. It may be corrupted or an unsupported format.")
    except IOError:
        print("An I/O error occurred while accessing the file.")
    except ValueError as ve:
        print(f"Error processing the file: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Call the function
pdf_text, page = read_pdf_file(pdf_file_path)
text = " ".join(pdf_text)

# Remove html tags
def remove_html_tags(text):
    # Regular expression to match HTML tags
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

# Removing HTML tags
cleaned_text = remove_html_tags(text)

# Removing URL
def remove_url(text):
    pattern = re.compile(r'http\S+')
    return pattern.sub(r'', text)

cleaned_text = remove_url(cleaned_text)
text = cleaned_text.lower()

# Basic tokenizer
def tokenize_text(text):
    # Remove punctuation, lower case, and split sentences
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    words = re.findall(r'\w+', text.lower())
    return sentences, words

# Removing common stopwords
def remove_stopwords(words, stopwords):
    return [word for word in words if word not in stopwords]

# Stemming (we can add more complex lemmatization if needed)
def stem_word(word):
    suffixes = ['ing', 'ly', 'ed', 'ious', 'ies', 'ive', 'es', 's', 'ment']
    for suffix in suffixes:
        if word.endswith(suffix):
            return word[:-len(suffix)]
    return word

def frequency_based_summarization(text, stopwords, summary_ratio):
    sentences, words = tokenize_text(text)
    words = remove_stopwords(words, stopwords)
    
    # Frequency of each word
    word_frequencies = Counter(words)
    
    # Scoring sentences
    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        sentence_words = re.findall(r'\w+', sentence.lower())
        
        for word in sentence_words:
            word = stem_word(word)

            if word in word_frequencies:
                if i not in sentence_scores:
                    sentence_scores[i] = word_frequencies[word]
                else:
                    sentence_scores[i] += word_frequencies[word]
    
    # Sort sentences by score
    summary_length = int(len(sentences) * summary_ratio)

    summary_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:summary_length]
    summary_sentences.sort()
    
    # Return the best scoring sentences as a summary
    return ' '.join([sentences[i] for i in summary_sentences])

stopwords = stopwords.words()

def adjust_based_on_length(text, stopwords, page):
    sentences, words = tokenize_text(text)
    
    # For short documents, summarize more briefly
    if page < 100:
        summary = frequency_based_summarization(text, stopwords, summary_ratio=0.1)
        
    # For medium-length documents
    elif page < 150:
        summary = frequency_based_summarization(text, stopwords, summary_ratio=0.01)
        
    # For long documents
    else:
        summary = frequency_based_summarization(text, stopwords, summary_ratio=0.05)
  
    return summary

def top_keywords(pdf_text):
    # Initialize the TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')

    # Fit the model and transform the documents into a matrix
    tfidf_matrix = vectorizer.fit_transform(pdf_text)

    # Get feature names (i.e., words)
    feature_names = vectorizer.get_feature_names_out()

    dic = {}
    for doc_idx, doc in enumerate(pdf_text):
        for word_idx in tfidf_matrix[doc_idx].nonzero()[1]:
            dic[feature_names[word_idx]] = tfidf_matrix[doc_idx, word_idx]

    sorted_by_values_desc = dict((list(sorted(dic.items(), key=lambda item: item, reverse=True)))[:20])

    key_words = list(sorted_by_values_desc.keys())
    
    return key_words

summary = adjust_based_on_length(text, stopwords, page)
key_words = top_keywords(pdf_text)

# Get file size
file_size = os.path.getsize(pdf_file_path)

# Prepare document for MongoDB
document = {
    "file_name": os.path.basename(pdf_file_path),
    "file_path": pdf_file_path,
    "file_size": file_size,
    "summary": summary,
    "keywords": key_words
}

# Insert document into MongoDB
collection.insert_one(document)

st.write(summary)
st.write(key_words)
