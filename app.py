import streamlit as st
import torch
import faiss
from sentence_transformers import SentenceTransformer
from transformers import BartForConditionalGeneration, BartTokenizer

# Initialize the sentence transformer model for semantic search
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the Hugging Face BART model and tokenizer for RAG
model_name = "facebook/bart-large-cnn"
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)

# Load the pre-generated embeddings and corpus (knowledge base)
embeddings = torch.load("knowledge_base_embeddings.pt")  # Ensure this is the correct file path

# Read the corpus (knowledge base text file)
with open("knowledge_base.txt", "r", encoding='utf-8') as file:
    corpus = file.readlines()

# Convert embeddings to numpy for FAISS
embeddings_np = embeddings.cpu().numpy()

# Initialize FAISS index for efficient search
index = faiss.IndexFlatL2(embeddings_np.shape[1])  
index.add(embeddings_np)

# Function to retrieve the top K relevant documents for a query
def get_relevant_documents(query, top_k=5):
    query_embedding = sentence_model.encode([query], convert_to_tensor=True)
    query_embedding_np = query_embedding.cpu().numpy()  
    D, I = index.search(query_embedding_np, top_k)  # Perform semantic search
    relevant_documents = [corpus[i] for i in I[0]]  # Retrieve the top K documents
    return relevant_documents

# Function to generate an answer using the BART model (RAG)
def generate_answer(query, relevant_docs):
    input_text = " ".join(relevant_docs) + " " + query
    inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True, padding="longest")
    summary_ids = model.generate(inputs["input_ids"], num_beams=4, min_length=50, max_length=500)
    answer = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return answer

# Streamlit app UI
st.title('AskLex10 Legal Question Answering System By Haritashva')
st.subheader('Ask a legal question, and get an answer based on the knowledge base.')

# Text input for the user query
query = st.text_input("Enter your legal question:")

if query:
    with st.spinner('Retrieving relevant documents and generating the answer...'):
        # Retrieve the top K relevant documents based on the query
        relevant_documents = get_relevant_documents(query, top_k=5)

        # Display the relevant documents retrieved
        st.subheader("Relevant Documents:")
        for i, doc in enumerate(relevant_documents):
            st.write(f"{i+1}. {doc[:300]}...")  # Display the first 300 characters of each relevant document

        # Generate the answer using the retrieved documents
        answer = generate_answer(query, relevant_documents)

        # Display the answer
        st.subheader("Answer:")
        st.write(answer)
