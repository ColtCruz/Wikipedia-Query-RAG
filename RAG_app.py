# ------------------------------
# Suppress TensorFlow/Keras
# ------------------------------
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

# ------------------------------
# Step 3.1: Logging & Warnings
# ------------------------------
import logging
import warnings
from transformers import pipeline, logging as hf_logging

logging.getLogger("langchain.text_splitter").setLevel(logging.ERROR)
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

# ------------------------------
# Step 3.2: Parameters
# ------------------------------
chunk_size = 500
chunk_overlap = 50
model_name = "sentence-transformers/all-distilroberta-v1"
top_k = 5

# ------------------------------
# Step 3.3: Read Document
# ------------------------------
with open("Selected_Document.txt", "r", encoding="utf-8") as f:
    text = f.read()

# ------------------------------
# Step 3.4: Chunking
# ------------------------------
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    separators=['\n\n', '\n', ' ', ''],
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)
chunks = splitter.split_text(text)

print(f"\n[INFO] Loaded {len(chunks)} chunks.")
print("[SAMPLE CHUNK]")
print(chunks[0][:300])  # Print the first 300 characters of the first chunk


# ------------------------------
# Step 3.5: Embeddings & FAISS
# ------------------------------
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

embedder = SentenceTransformer(model_name)
embeddings = embedder.encode(chunks, show_progress_bar=True)
embedding_array = np.array(embeddings).astype("float32")

index = faiss.IndexFlatL2(embedding_array.shape[1])
index.add(embedding_array)

# ------------------------------
# Step 3.6: Generator
# ------------------------------
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    device=-1
)

# ------------------------------
# Step 3.7: Retrieval & Answer
# ------------------------------
def retrieve_chunks(question, k=top_k):
    question_embedding = embedder.encode([question]).astype("float32")
    distances, indices = index.search(question_embedding, k)
    return [chunks[idx] for idx in indices[0]]

def answer_question(question):
    retrieved = retrieve_chunks(question)
    context = "\n".join(retrieved)
    prompt = f"Answer the question based on the context below.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    response = generator(prompt, max_length=256, do_sample=False)
    return response[0]["generated_text"]

# ------------------------------
# Step 3.8: Interactive Loop
# ------------------------------
if __name__ == "__main__":
    print("Enter 'exit' or 'quit' to end.")
    while True:
        question = input("Your question: ")
        if question.lower() in ("exit", "quit"):
            break
        print("Answer:", answer_question(question))
