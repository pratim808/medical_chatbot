from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from src.helper import load_pdf, text_split, download_hugging_face_embeddings
DATA_PATH = r'G:\Chatbot\data'
DB_FAISS_PATH = r'G:\Chatbot\data\vector'


extracted_data = load_pdf(DATA_PATH)
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()
db = FAISS.from_documents(text_chunks, embeddings)
db.save_local(DB_FAISS_PATH)
print("### db is created")



