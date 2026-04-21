# INDEX PDF FILES  
# 1. build file scanner
import re
import os
root_path = os.path.dirname(os.path.abspath(__file__))
input_path = f"{root_path}/pdf_test"

def scan_files(input_path):
    file_list = []
    num_files = 0
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.endswith(".pdf"):
                file_list.append(os.path.join(root, file))
                num_files += 1
    return file_list, num_files

file_list, num_files = scan_files(input_path)
print(f"Files found: {num_files}")
print(file_list)

# 2. open and validate file list
for file in file_list:
    try:
        with open(file, 'rb') as f:
            print(f"Successfully opened: {file}")
    except Exception as e:
        print(f"Error opening {file}: {e}")


# EXTRACT TEXT
# 1. ẽxtract text from 1 pdf file
from PyPDF2 import PdfReader
def extract_text_from_pdf(file_path):
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error extracting text from {file_path}: {e}")
        return None

def clean_text(text):
    # fix hyphen line breaks
    text = text.replace("-\n", "")

    # remove dotted leaders
    text = re.sub(r"\.{2,}", " ", text)

    # remove trailing page numbers
    text = re.sub(r"\s+\d+\s*$", "", text, flags=re.MULTILINE)

    # remove non-breaking spaces early
    text = text.replace("\u00a0", " ")

    # replace newlines with space
    text = text.replace("\n", " ")

    # collapse all whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()
def chunk_text(text, chunk_size=400, overlap=100):
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    
    for i in range(0, len(words), step):
        chunk_words = words[i:i+chunk_size]
        if len(chunk_words) < 50:
            continue
        chunk_text = " ".join(chunk_words)
        chunks.append(chunk_text)
    
    return chunks
list_of_chunk_text = []
elements = {} # "file": name, "chunk_id": id, "text": text

for file in file_list:
    text = extract_text_from_pdf(file)
    if text:
        text = clean_text(text)
        chunks = chunk_text(text)
        print(f"Cleaned text from {file} successfully.")
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i+1}: {chunk[:100]}")
            list_of_chunk_text.append({
                "file": file,
                "chunk_id": i,
                "text": chunk
            })
    else:
        print(f"Failed to extract text from {file}.")
            

"""
INPUT = 
        [
            {
                "file": "lab1.pdf",
                "chunk_id": 0,
                "text": "Operating Systems..."
            }
        ]

OUTPUT =
A. Vector index (FAISS)
        [
            [0.12, -0.98, ...],  # vector 0
            [0.33,  0.44, ...],  # vector 1
        ]

B. Metadata index
        {
            0: {"file": "lab1.pdf", "chunk_id": 0},
            1: {"file": "lab1.pdf", "chunk_id": 1}
        }
"""
# CHOOSE EMBEDDING MODEL
from sentence_transformers import SentenceTransformer
model_name = "all-MiniLM-L6-v2" # fast, light weight
# load model
model = SentenceTransformer(model_name)
print(f"Loaded embedding model: {model_name}")

batch_size = 32 
vectors = model.encode([item["text"] for item in list_of_chunk_text], batch_size=batch_size, show_progress_bar=True)

# store metadata
# id = 0, 1, 2, 3...
metadata_index = {}
for i, item in enumerate(list_of_chunk_text):
    metadata_index[i] = {
        "file": item["file"],
        # unique chunk id within the file
        "chunk_id": item["chunk_id"]
}

import faiss
# use IndexFlatL2 for simplicity, but can use more advanced indexes for larger datasets
dimension = vectors.shape[1]  # dimension of the embedding vectors
index = faiss.IndexFlatL2(dimension)    
# add vectors to FAISS index
index.add(vectors)

#store FAISS index to disk
faiss_index_path = f"{root_path}/faiss_index.bin"
faiss.write_index(index, faiss_index_path)
print(f"Saved FAISS index to {faiss_index_path}")

# store metadata index to disk
import json
metadata_path = f"{root_path}/metadata_index.json"
with open(metadata_path, 'w') as f:
    json.dump(metadata_index, f)
print(f"Saved metadata index to {metadata_path}")

"""
    Choice 1: Normalize vectors?
    Choice 2: Distance metric
    Choice 3: top_k
"""