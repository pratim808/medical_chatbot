from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings

from langchain_community.vectorstores import FAISS

from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
# from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
from store_index import create_vector_db
app = Flask(__name__)

# load_dotenv()

# PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
# PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')


# DATA_PATH = '/kaggle/input/book-pdf'
# DB_FAISS_PATH = r'G:\Chatbot\data\vector'

# Create vector database
create_vector_db()
print("###333")
# Call the function directly in the cell

'''#Initializing the Pinecone
pinecone.init(api_key=PINECONE_API_KEY,
              environment=PINECONE_API_ENV)

index_name="medical-bot"'''

embeddings = download_hugging_face_embeddings()
# Load the FAISS vector database

# also i change here
# db = FAISS.load_local(DB_FAISS_PATH, embeddings)
DB_FAISS_PATH = r'G:\Chatbot\data\vector'
print("vector_base is loading from the folder")
db = FAISS.load_local(DB_FAISS_PATH, embeddings,
                      allow_dangerous_deserialization=True)
# Loading the index
# db = FAISS.load_local(r"G:\Chatbot\DB_FAISS_PATH",embeddings, allow_dangerous_deserialization=True)


# docsearch=Pinecone.from_existing_index(index_name, embeddings)

# Create a ChatPromptTemplate
prompt = ChatPromptTemplate.from_template(prompt_template)

chain_type_kwargs = {"prompt": prompt}

'''PROMPT = PromptTemplate(template=prompt_template,
                        input_variables=["context", "question"])

chain_type_kwargs = {"prompt": PROMPT}'''

'''llm = CTransformers(model=r"G:\Chatbot\model\llama-2-7b-chat.ggmlv3.q4_0.bin",
                    model_type="llama",
                    config={'max_new_tokens': 512,
                            'temperature': 0.8})'''

# Initialize the LLM
# llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
# groq_api_key = ('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')
# Load the GROQ and OpenAI API keys
groq_api_key = os.getenv('GROQ_API_KEY')
# os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
# Initialize the LLM
llm = ChatGroq(groq_api_key=groq_api_key,
               model_name="Llama3-8b-8192")


qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs)


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result = qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
